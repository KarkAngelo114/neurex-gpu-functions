#include <napi.h>
#include <vector>
#include <string>
#include "../gpu/gpu_context.h"
#include "globals.h"
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

static std::string compute_backend_string;

namespace {
struct BufferStore {
    napi_env env;
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, Napi::Reference<Napi::Float32Array>>> buffers;
};

std::mutex buffer_stores_mutex;
std::unordered_map<napi_env, BufferStore*> buffer_stores;

void cleanupBufferStore(void* data) {
    auto* store = static_cast<BufferStore*>(data);
    {
        std::lock_guard<std::mutex> lock(buffer_stores_mutex);
        buffer_stores.erase(store->env);
    }
    delete store;
}

BufferStore& getBufferStore(Napi::Env env) {
    const napi_env rawEnv = env;
    std::lock_guard<std::mutex> lock(buffer_stores_mutex);

    auto found = buffer_stores.find(rawEnv);
    if (found != buffer_stores.end()) {
        return *found->second;
    }

    auto* store = new BufferStore{rawEnv, {}};
    try {
        env.AddCleanupHook(cleanupBufferStore, store);
    } catch (...) {
        delete store;
        throw;
    }
    buffer_stores.emplace(rawEnv, store);
    return *store;
}
}


std::string getComputeBackendType() {
    return compute_backend_string;
}

Napi::Float32Array getOrCreateBuffer(Napi::Env env, const std::string& global_reference_id, const std::string& local_reference_id, int size) {
    if (global_reference_id.empty() || local_reference_id.empty()) {
        throw std::invalid_argument("Buffer reference IDs must not be empty.");
    }
    if (size < 0) {
        throw std::invalid_argument("Buffer size must not be negative.");
    }

    auto& buffers = getBufferStore(env).buffers[global_reference_id];
    auto found = buffers.find(local_reference_id);
    if (found != buffers.end() &&
        found->second.Value().As<Napi::Float32Array>().ElementLength() == static_cast<size_t>(size)) {
        return found->second.Value();
    }

    Napi::Float32Array buffer = Napi::Float32Array::New(env, static_cast<size_t>(size));
    buffers.insert_or_assign(local_reference_id, Napi::Persistent(buffer));
    return buffer;
}

Napi::Value setComputeBackendType(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    compute_backend_string = info[0].As<Napi::String>().Utf8Value();
    return env.Undefined();
}

Napi::Value UploadParamsFromJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsArray() || !info[2].IsArray()) {
        Napi::TypeError::New(env, "Expected: modelID (string), weights (array), biases (array).").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string modelID = info[0].As<Napi::String>().Utf8Value();
    Napi::Array jsWeights = info[1].As<Napi::Array>();
    Napi::Array jsBiases = info[2].As<Napi::Array>();
    std::string computeBackend = info[3].As<Napi::String>().Utf8Value();

    Matrix weightMatrix(jsWeights.Length());
    Matrix biasMatrix(jsBiases.Length());

    for (uint32_t i = 0; i < jsWeights.Length(); i++) {
        Napi::Float32Array w = jsWeights.Get(i).As<Napi::Float32Array>();
        weightMatrix[i].assign(w.Data(), w.Data() + w.ElementLength());
    }

    for (uint32_t i = 0; i < jsBiases.Length(); i++) {
        Napi::Float32Array b = jsBiases.Get(i).As<Napi::Float32Array>();
        biasMatrix[i].assign(b.Data(), b.Data() + b.ElementLength());
    }


    // if "opencl", upload params to OpenCL
    if (computeBackend == "opencl") {
         auto& openCL = GpuContext::instance();

        std::string err;
        bool success = openCL.uploadParams(modelID, weightMatrix, biasMatrix, err);

        if (!success) {
            Napi::Error::New(env, err).ThrowAsJavaScriptException();
        }

        
    }

    return env.Undefined();
}

Napi::Value ReleaseParams(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected: modelID (string).").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string modelID = info[0].As<Napi::String>().Utf8Value();

    if (compute_backend_string == "opencl") {
        auto& gpu = GpuContext::instance();
        gpu.clearParams(modelID);
        // clearParams() only releases weights/biases now (see gpu_context.cpp for why).
        // ReleaseParams is JS's explicit "I'm done with this model" signal, so it's the
        // right place to also drop the model's cached optimizer states and activation buffers.
        gpu.clearOptimizerStates(modelID);
        gpu.clearActivationCaches(modelID);
        gpu.clear_dBeta_And_dGamma_By_Model(modelID);

        std::cout << "> CLbuffers has been cleared." << std::endl;
        
    }

    
    return env.Undefined();
    
}

Napi::Value shutdownGPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (compute_backend_string == "opencl") {
        auto& OpenCL = GpuContext::instance();
        bool res = OpenCL.shutdown();

        if (res) {
            std::cout << "> clBuffers cleared. \n> kernel source cleared. \n> Memory released." << std::endl;
        }
        else {
            Napi::Error::New(env, "Failed to shutdown. \nIf this error occurred, please open an issue to: https://github.com/KarkAngelo114/Neurex/issues").ThrowAsJavaScriptException();
        }

        
    }

    return env.Undefined();
    
}

void _globals(Napi::Env env, Napi::Object exports) {
    exports.Set("setComputeBackendType", Napi::Function::New(env, setComputeBackendType));
    exports.Set("UploadParams", Napi::Function::New(env, UploadParamsFromJS));
    exports.Set("ReleaseParams", Napi::Function::New(env, ReleaseParams));
    exports.Set("shutdown", Napi::Function::New(env, shutdownGPU));
}