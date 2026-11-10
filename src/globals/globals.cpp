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

    if (info.Length() < 3 || !info[0].IsString() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "Expected: modelID (string), layerParameters (array), computeBackend (string).").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string modelID = info[0].As<Napi::String>().Utf8Value();
    Napi::Array jsParameters = info[1].As<Napi::Array>();
    std::string computeBackend = info[2].As<Napi::String>().Utf8Value();
    std::vector<LayerParameters> parameters;
    parameters.reserve(jsParameters.Length());

    for (uint32_t i = 0; i < jsParameters.Length(); i++) {
        Napi::Value value = jsParameters.Get(i);
        if (!value.IsObject()) {
            Napi::TypeError::New(env, "Each layer parameter entry must be an object.").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        Napi::Object layer = value.As<Napi::Object>();
        Napi::Value idValue = layer.Get("layerID");
        Napi::Value weightsValue = layer.Get("weights");
        Napi::Value biasesValue = layer.Get("biases");
        if (!idValue.IsString() || !weightsValue.IsTypedArray() || !biasesValue.IsTypedArray() ||
            weightsValue.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array ||
            biasesValue.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
            Napi::TypeError::New(env, "Layer parameters require layerID and Float32Array weights and biases.").ThrowAsJavaScriptException();
            return env.Undefined();
        }

        Napi::Float32Array weights = weightsValue.As<Napi::Float32Array>();
        Napi::Float32Array biases = biasesValue.As<Napi::Float32Array>();
        parameters.push_back({
            idValue.As<Napi::String>().Utf8Value(),
            std::vector<float>(weights.Data(), weights.Data() + weights.ElementLength()),
            std::vector<float>(biases.Data(), biases.Data() + biases.ElementLength())
        });
    }


    // if "opencl", upload params to OpenCL
    if (computeBackend == "opencl") {
         auto& openCL = GpuContext::instance();

        std::string err;
        bool success = openCL.uploadParams(modelID, parameters, err);

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