// src/init.cpp
#include <napi.h>
#include "gpu/gpu_context.h"

static Napi::Value InitGPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Object out = Napi::Object::New(env);

    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsNumber()) {
        out.Set("ok", Napi::Boolean::New(env, false));
        out.Set("error", Napi::String::New(env, "Init_GPU(kernelPath: string, deviceIndex: number) expected"));
        return out;
    }

    std::string kernelPath = info[0].As<Napi::String>().Utf8Value();
    uint32_t deviceIndex   = info[1].As<Napi::Number>().Uint32Value();
    std::string err;

    bool ok = GpuContext::instance().initialize(kernelPath, deviceIndex, err);

    out.Set("ok", Napi::Boolean::New(env, ok));
    out.Set("error", Napi::String::New(env, err));
    return out;
}

static Napi::Value ShutdownGPU(const Napi::CallbackInfo& info) {
    GpuContext::instance().shutdown();
    return info.Env().Undefined();
}

void GpuLifecycleRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("Init_GPU",     Napi::Function::New(env, InitGPU));
    exports.Set("Shutdown_GPU", Napi::Function::New(env, ShutdownGPU));
}