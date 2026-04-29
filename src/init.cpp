// src/init.cpp
#include <napi.h>
#include "gpu/gpu_context.h"

static Napi::Value InitGPU(const Napi::CallbackInfo& info) {
    std::string err;
    bool ok = GpuContext::instance().initialize(err);

    Napi::Object out = Napi::Object::New(info.Env());
    
    out.Set("ok", Napi::Boolean::New(info.Env(), ok));
    out.Set("error", Napi::String::New(info.Env(), err));
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