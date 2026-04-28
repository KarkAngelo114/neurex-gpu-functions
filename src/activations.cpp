/*
    This is the native code of activations and their derivatives
    - Relu
    - Sigmoid
    - Tanh
    - Softmax
    - Linear

    - ReLu Derivative
    - Sigmoid Derivative
    - Tanh Derivative
    - Linear Derivative
    - Softmax Derivative

    All functions returns a 1D array

*/

#include <napi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;


/* ========================= Callable functions ============================*/
Napi::Value ReluWrapper(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
    return input;
}

Napi::Value SigmoidWrapper(const Napi::CallbackInfo& info) {
   Napi::Env env = info.Env();
   Napi::Float32Array input = info[0].As<Napi::Float32Array>();

   float* data = input.Data();
   size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = 1.0f / (1.0f + exp(-data[i]));
    }

   return input;
}

Napi::Value TanhWrapper(const Napi::CallbackInfo& info) {
   Napi::Env env = info.Env();
   Napi::Float32Array input = info[0].As<Napi::Float32Array>();

   float* data = input.Data();
   size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = tanh(data[i]);
    }

   return input;
}

Napi::Value SoftmaxWrapper(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    float max_val = data[0];
    for (size_t i = 1; i < input_size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input_size; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    for (size_t i = 0; i < input_size; i++) {
        data[i] /= sum;
    }

    return input;
}

Napi::Value LinearWrapper(const Napi::CallbackInfo& info) {
    return info[0];
}

/* ========================= Derivatives ============================*/
Napi::Value DReLuWrapper(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
    }
    return input;
}

Napi::Value DSigmoidWrapper(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        float s = 1.0f / (1.0f + std::exp(-data[i]));
        data[i] = s * (1.0f - s);
    }
    return input;
}

Napi::Value DTanhWrapper(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        float t = std::tanh(data[i]);
        data[i] = 1.0f - (t * t);
    }
    return input;
}

Napi::Value DSoftmaxWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int arr_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_size);

    std::fill(output.Data(), output.Data() + arr_size, 1.0f);

    return output;
}

Napi::Value DLinearWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t arr_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_size);

    std::fill(output.Data(), output.Data() + arr_size, 1.0f);

    return output;
}

void ActivationsRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("Relu", Napi::Function::New(env, ReluWrapper));
    exports.Set("Sigmoid", Napi::Function::New(env, SigmoidWrapper));
    exports.Set("Tanh", Napi::Function::New(env, TanhWrapper));
    exports.Set("Softmax", Napi::Function::New(env, SoftmaxWrapper));
    exports.Set("Linear", Napi::Function::New(env, LinearWrapper));
    exports.Set("DReLu", Napi::Function::New(env, DReLuWrapper));
    exports.Set("DSigmoid", Napi::Function::New(env, DSigmoidWrapper));
    exports.Set("DTanh", Napi::Function::New(env, DTanhWrapper));
    exports.Set("DSoftmax", Napi::Function::New(env, DSoftmaxWrapper));
    exports.Set("DLinear", Napi::Function::New(env, DLinearWrapper));
}