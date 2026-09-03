#include <napi.h>
#include <vector>
#include <string>
#include "../gpu/gpu_context.h"
#include "globals.h"
#include <iostream>

static bool global_boolean_On_GPU_state;

bool get_Global_Boolean_On_GPU() {
    return global_boolean_On_GPU_state;
}

Napi::Value setOnGPU_Boolean_State(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    global_boolean_On_GPU_state = info[0].As<Napi::Boolean>().Value();
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

    auto& openCL = GpuContext::instance();

    std::string err;
    bool success = openCL.uploadParams(modelID, weightMatrix, biasMatrix, err);

    if (!success) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
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
    GpuContext::instance().clearParams(modelID);

    return env.Undefined();
}

Napi::Value shutdownGPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto& OpenCL = GpuContext::instance();
    bool res = OpenCL.shutdown();

    if (res) {
        std::cout << "> clBuffers cleared. \n> kernel source cleared. \n> Memory released." << std::endl;
    }
    else {
        Napi::Error::New(env, "Failed to shutdown. \nIf this error occurred, please open an issue to: https://github.com/KarkAngelo114/Neurex/issues").ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

void _globals(Napi::Env env, Napi::Object exports) {
    exports.Set("setOnGPU", Napi::Function::New(env, setOnGPU_Boolean_State));
    exports.Set("UploadParams", Napi::Function::New(env, UploadParamsFromJS));
    exports.Set("ReleaseParams", Napi::Function::New(env, ReleaseParams));
    exports.Set("shutdown", Napi::Function::New(env, shutdownGPU));
}