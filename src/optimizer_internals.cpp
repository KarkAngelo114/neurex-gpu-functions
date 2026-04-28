#include <napi.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


Napi::Value SGD_wrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    float lr = info[2].As<Napi::Number>().FloatValue();

    float* p = params.Data();
    float* g = grads.Data();
    size_t element_length = params.ElementLength();

    for (size_t i = 0; i < element_length; i++) {
        p[i] -= lr * g[i];
    }

    return params;

    
}

Napi::Value Adam_wrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array stateM = info[2].As<Napi::Float32Array>();
    Napi::Float32Array stateV = info[3].As<Napi::Float32Array>();
    int stateT = info[4].As<Napi::Number>().Int32Value();
    float learning_rate = info[5].As<Napi::Number>().FloatValue();
    float beta1 = info[6].As<Napi::Number>().FloatValue();
    float beta2 = info[7].As<Napi::Number>().FloatValue();
    float epsilon = info[8].As<Napi::Number>().FloatValue();

    float* p = params.Data();
    float* g = grads.Data();
    float* sm = stateM.Data();
    float* sv = stateV.Data();
    size_t params_len = params.ElementLength();

    for (size_t i = 0; i < params_len; i++) {
        float grad = g[i];

        sm[i] = beta1 * sm[i] + (1 - beta1) * grad;
        sv[i] = beta2 * sv[i] + (1 - beta2) * grad * grad;

        float mHat = sm[i] / (1.0f - pow(beta1, stateT));
        float vHat = sv[i] / (1.0f - pow(beta2, stateT));

        p[i] -= learning_rate * mHat / (sqrt(vHat) + epsilon);
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("params", params);
    output.Set("m", stateM);
    output.Set("v", stateV);

    return output;
}

// ======== exports ======== //
void OptimizerInternals(Napi::Env env, Napi::Object exports) {
    exports.Set("SGD", Napi::Function::New(env, SGD_wrapper));
    exports.Set("Adam", Napi::Function::New(env, Adam_wrapper));
}