#include <iostream>
#include <napi.h>
#include <vector>
using namespace std;

Napi::Value ScaleGradsWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array grads = info[0].As<Napi::Float32Array>();
    int batchSize = info[1].As<Napi::Number>().Int32Value();

    float* data = grads.Data();
    size_t length = grads.ElementLength();

    for (size_t i = 0; i < length; i++) {
        data[i] /= batchSize;
    }

    return grads;
}


void ScaleGradients(const Napi::Env env, const Napi::Object exports) {
   exports.Set("scaleGrad", Napi::Function::New(env, ScaleGradsWrapper));
}