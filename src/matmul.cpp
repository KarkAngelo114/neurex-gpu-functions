// #include <napi.h>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;


// Napi::Value MatMulWrapper(const Napi::CallbackInfo& info) {
//     Napi::Env env = info.Env();
//     Napi::Float32Array input = info[0].As<Napi::Float32Array>();
//     Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
//     Napi::Float32Array biases = info[2].As<Napi::Float32Array>();
//     int inputSize = info[3].As<Napi::Number>().Int32Value();
//     int outputSize = info[4].As<Napi::Number>().Int32Value();

//     Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);

//     float* inputData = input.Data();
//     float* w = weights.Data();
//     float* b = biases.Data();
//     float* x = output.Data();

//     std::copy(b, b + outputSize, x);

//     for (size_t i = 0; i < inputSize; i++) {
//         float inputVal = inputData[i];
//         int offset = i * outputSize;

//         for (size_t j = 0; j < outputSize; j++) {
//             x[j] += inputVal * w[offset + j];
//         }
//     }

//     return output;
// }

// Napi::Value DeltaMatMulWrapper(const Napi::CallbackInfo& info) {
//     Napi::Env env = info.Env();
//     Napi::Float32Array delta = info[0].As<Napi::Float32Array>();
//     Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
//     int inputSize = info[2].As<Napi::Number>().Int32Value();
//     int outputSize = info[3].As<Napi::Number>().Int32Value();

//     Napi::Float32Array output = Napi::Float32Array::New(env, inputSize);

//     float* d = delta.Data();
//     float* w = weights.Data();
//     float* o = output.Data();

//     for (size_t i = 0; i < inputSize; i++) {
//         float sum = 0.0f;
//         int offset = i * outputSize;

//         for (size_t j = 0; j < outputSize; j++) {
//             sum += w[offset + j]  * d[j];
//         }
//         o[i] = sum;
//     }

//     return output;
// }

#include <napi.h>
#include <Eigen/Dense>

using RowMatXf  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstMapM = Eigen::Map<const RowMatXf>;
using ConstMapV = Eigen::Map<const Eigen::VectorXf>;
using MapV      = Eigen::Map<Eigen::VectorXf>;
using ConstMapRV= Eigen::Map<const Eigen::RowVectorXf>;
using MapRV     = Eigen::Map<Eigen::RowVectorXf>;

Napi::Value MatMulWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto input   = info[0].As<Napi::Float32Array>();
    auto weights = info[1].As<Napi::Float32Array>();
    auto biases  = info[2].As<Napi::Float32Array>();
    int  inSize  = info[3].As<Napi::Number>().Int32Value();
    int  outSize = info[4].As<Napi::Number>().Int32Value();

    auto output = Napi::Float32Array::New(env, outSize);

    ConstMapRV x(input.Data(),   inSize);
    ConstMapM  W(weights.Data(), inSize, outSize);
    ConstMapRV b(biases.Data(),  outSize);
    MapRV      y(output.Data(),  outSize);

    y.noalias() = x * W;
    y += b;
    return output;
}

Napi::Value DeltaMatMulWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto delta   = info[0].As<Napi::Float32Array>();
    auto weights = info[1].As<Napi::Float32Array>();
    int  inSize  = info[2].As<Napi::Number>().Int32Value();
    int  outSize = info[3].As<Napi::Number>().Int32Value();

    auto output = Napi::Float32Array::New(env, inSize);

    ConstMapM W(weights.Data(), inSize, outSize);
    ConstMapV d(delta.Data(), outSize);
    MapV o(output.Data(), inSize);

    o.noalias() = W * d;
    return output;
}

// =================== MODULE EXPORT ===================
void MatMulRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("MatMul", Napi::Function::New(env, MatMulWrapper));
    exports.Set("DeltaMatMul", Napi::Function::New(env, DeltaMatMulWrapper));
}