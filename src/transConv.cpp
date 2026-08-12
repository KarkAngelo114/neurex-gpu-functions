#include <napi.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <vector>
#include <cmath>
using IntArray = sts::vector<int>;


static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}


Napi::Value transConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[7].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, oH * oW * f);

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    // helper functions
    auto inputIndex = [&](int y, int x, int c) {
        return (y * iW + x) * iD + c;
    };
    auto outputIndex = [&](int y, int x, int c) {
        return (y * oW + x) * f + c;
    };
    auto weightIndex = [&](int filter, int ky, int kx, int c) {
        return (((filter * kh) + ky) * kw + kx) * d + c;
    };

    float* input = inputTensor.Data();
    float* weights = weightsArray.Data();
    float* biases = biasesArray.Data(); 
    float* output = outputTensor.Data();
    

    // set biases before hand
    for (int y = 0; y < oH; y++) {
        for (int x = 0; x < oW; x++) {

            int outputBase = (y * oW + x) * f;

            for (int filter = 0; filter < f; filter++) {
                output[outputBase + filter] = biases[filter]; // for safety, we use "=" rather "+="
            }
        }
    }

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {

            int inputBase = (iy * iW + ix) * iD;

            for (int ky = 0; ky < kh; ky++) {

                int oy = iy * strides + ky - padTop;

                // Kernel row falls outside output.
                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {

                    int ox = ix * strides + kx - padLeft;

                    // Kernel column falls outside output.
                    if (ox < 0 || ox >= oW) continue;

                    int outputBase = (oy * oW + ox) * f;

                    /*
                     * For every output filter, accumulate the
                     * input channels multiplied by the kernel.
                     */
                    for (int filter = 0; filter < f; filter++) {

                        float sum = 0.0f;

                        int weightBase = ((filter * kh + ky) * kw + kx) * d;

                        for (int c = 0; c < d; c++) {
                            sum += input[inputBase + c] * weights[weightBase + c];
                        }

                        output[outputBase + filter] += sum; // we use "+=" to sum to the bias
                    }
                }
            }
        }
    }

    return outputTensor;
}

Napi::Value transConvBackward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, iH * iW * iD);

    // helper function
    auto deltaOutputIndex = [&](int y, int x, int f) {
        return (y * oW + x) * oD + f;
    };

    float* delta = deltaTensor.Data();
    float* weights = weightsArray.Data();
    float* deltaInput = outputTensor.Data();

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {
            for (int ky = 0; ky < kh; ky++) {
                int oy = iy * strides + ky - padTop;

                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {

                    int ox = ix * strides + kx - padLeft;

                    if (ox < 0 || ox >= oW) continue;

                    for (int filter = 0; filter < f; filter++) {

                        float deltaY = delta[deltaOutputIndex(oy, ox, filter)];
                        int weightBase = ((filter * kh + ky) * kw + kx) * d;
                        int inputBase = (iy * iW + ix) * iD;

                        for (int c = 0; c < d; c++) {

                            deltaInput[inputBase + c] += deltaY * weights[weightBase + c];
                        }
                    }
                }
            }
        }
    }

    return outputTensor;
}

Napi::Value transConvWrapper(const Napi::CallbackInfo& info) {
    return transConv_CPU(info);
}

Napi::Value transConvBackwardWrapper(const Napi::CallbackInfo& info) {
    return transConvBackward_CPU(info);
}

void transConvFunc(Napi::Env env, Napi::Object exports) {
    exports.Set("transConv", Napi::Function::New(env, transConvWrapper));
    exports.Set("transConvBackward", Napi::Function::New(env, transConvBackwardWrapper))
}