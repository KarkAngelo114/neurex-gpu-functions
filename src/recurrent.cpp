#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "functions/functions.h"
#include <vector>

using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray Vector;
    Vector.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        Vector.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return Vector;
}


Napi::Value recurrentMatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_data = info[0].As<Napi::Float32Array>();
    Napi::Float32Arrray prevHiddenState_data = info[1].As<Napi::Float32Array>();
    IntArray inputWeightShape = Vectorize(info[2].As<Napi::Array>());
    IntArray recurrentWeightShape = Vectorize(info[3].As<Napi::Array>());
    Napi::Float32Array weight_data = info[4].As<Napi::Float32Array>(); // [...inputWeights, ...recurrentWeights]
    Napi::Float32Array biases_array = info[5].As<Napi::Float32Array>();
    

    int inputSize = inputWeightShape[0];
    int units = inputWeightShape[1];
    int range_input_weights = inputSize * units;

    Napi::Float32Array output_data = Napi::Float32Array::New(env, units);
    Napi::Float32Array input_weights_data = subarray(env, weight_data, 0, range_input_weights);
    Napi::Float32Array recurrent_weights_data = subarray(env, weight_data, range_input_weights, range_input_weights + recurrentWeightShape[0] * recurrentWeightShape[1])

    float* input = input_data.Data();
    float* prevHiddenState = prevHiddenState_data.Data();
    float* input_weights = input_weight_data.Data();
    float* recurrent_weights = recurrent_weights_data.Data();
    float* biases = biases_array.Data();
    float* output = output_data.Data();

    for (size_t j = 0; j < units; j++) {
        float z = biases[j];

        for (size_t i = 0; i < inputSize; i++) {
            z += input[i] * input_weights[i * units + j];
        }

        for (size_t h = 0; h < units; h++) {
            z += prevHiddenState[h] * recurrent_weights[h * units + j];
        }

        output[j] = z;
    }
    
    return output_data;

}

Napi::Value recurrentTimeDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array delta_array = info[0].As<Napi::Float32Array>();
    IntArray inputWeightShape = Vectorize(info[2].As<Napi::Array>());
    IntArray recurrentWeightShape = Vectorize(info[3].As<Napi::Array>());
    Napi::Float32Array weight_data = info[4].As<Napi::Float32Array>(); // [...inputWeights, ...recurrentWeights]

    int a = inputWeightShape[0];
    int b = inputWeightShape[1];
    int c = recurrentWeightShape[0];
    int d = recurrentWeightShape[1];

    int offset = a * b;
    int length = c * d;

    Napi::Float32Array weights_array = subarray(env, weight_data, offset, offset + length); // get the recurrent weights
    Napi::Float32Array output_data = Napi::Float32Array::New(env, delta_array.ElementLength());

    float* delta = delta_array.Data();
    float* weights = weights_array.Data();
    float* output = output_data.Data();

    for (size_t i = 0; i < c; i++) {
        float sum = 0.0f;
        int offset = i * c;

        for (size_t j = 0; j < d; j++) {

            sum += weights[offset + j]  * delta[j];
        }

        output[i] = sum;
    }

    return output_data;

}


Napi::Value recurrentMatMulWrapper(const Napi::CallbackInfo& info) {
    return recurrentMatMul_CPU(info);
}

Napi::Value recurrentTimeDeltaWrapper(const Napi::CallbackInfo& info) {
    return recurrentTimeDelta_CPU(info);
}

void recurrentFunc(const Napi::Env env, const Napi::Object exports) {
    exports.Set("recurrentMatMul", Napi::Function::New(env, recurrentMatMulWrapper));
    exports.Set("recurrentTimeDelta", Napi::Function::New(env, recurrentTimeDeltaWrapper));
}