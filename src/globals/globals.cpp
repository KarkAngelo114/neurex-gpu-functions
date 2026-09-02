#include <napi.h>
#include <vector>
#include "../gpu/gpu_context.h"
#include "globals.h"
#include <iostream>

static bool global_boolean_On_GPU_state;

// get the boolean state initiated by JS
bool get_Global_Boolean_On_GPU() {
    return global_boolean_On_GPU_state;
}




Napi::Value setOnGPU_Boolean_State(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    global_boolean_On_GPU_state = info[0].As<Napi::Boolean>().Value();

    return env.Undefined();
}

Napi::Value UploadParams(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "Expected two arrays: weights and biases.").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array jsWeights = info[0].As<Napi::Array>();
    Napi::Array jsBiases = info[1].As<Napi::Array>();

    Matrix weightMatrix(jsWeights.Length());
    Matrix biasMatrix(jsBiases.Length());

    // Parse weights array of Float32Arrays
    for (uint32_t i = 0; i < jsWeights.Length(); i++) {
        Napi::Float32Array w = jsWeights.Get(i).As<Napi::Float32Array>();
        weightMatrix[i].assign(w.Data(), w.Data() + w.ElementLength());
    }

    // Parse biases array of Float32Arrays
    for (uint32_t i = 0; i < jsBiases.Length(); i++) {
        Napi::Float32Array b = jsBiases.Get(i).As<Napi::Float32Array>();
        biasMatrix[i].assign(b.Data(), b.Data() + b.ElementLength());
    }

    auto& openCL = GpuContext::instance();

    std::string err;
    bool success = openCL.uploadParams(weightMatrix, biasMatrix, err);

    if (!success) {
        Napi::Error::New(env, err).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value shutdownGPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    auto& OpenCL = GpuContext::instance();

    bool res = OpenCL.shutdown();

    if (res) {
        std::string successMessage = "> clBuffers cleared. \n> kernel source cleared. \n> Memory released.";
        std::cout << successMessage << std::endl;
    }
    else {
        std::string errorMessage = "Failed to shutdown. \nIf this error occurred, please open an issue to: https://github.com/KarkAngelo114/Neurex/issues";
        Napi::Error::New(env, errorMessage).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}


void _globals(Napi::Env env, Napi::Object exports) {
    exports.Set("setOnGPU", Napi::Function::New(env, setOnGPU_Boolean_State));
    exports.Set("UploadParams", Napi::Function::New(env, UploadParams));
    exports.Set("shutdown", Napi::Function::New(env, shutdownGPU));
}