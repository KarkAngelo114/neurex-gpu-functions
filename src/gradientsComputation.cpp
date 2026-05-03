#include <napi.h>
#include <vector>
#include <cmath>

Napi::Value ComputeGradientForDenseWeightsWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array activation_output = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputSize = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    float* a = activation_output.Data();
    float* d = deltas.Data();
    float* wg = weightGrads.Data();

    for (size_t i = 0; i < inputSize; i++) {
        float inputVal = a[i];
        int offset = i * outputSize;
        for (size_t j = 0; j < outputSize; j++) {
            wg[offset + j] += inputVal * d[j];
        }
    }
    return weightGrads;
}

Napi::Value computeBiasGradsForConnected_LayerWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasgrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[0].As<Napi::Float32Array>();

    float* bg = biasgrads.Data();
    float* d = deltas.Data();
    size_t length = biasgrads.ElementLength();

    for (size_t i = 0; i < length; i++) {
        bg[i] += d[i];
    }

    return biasgrads;
}

Napi::Value computeKernelGradients(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array delta = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputH = info[3].As<Napi::Number>().Int32Value(); 
    int inputW = info[4].As<Napi::Number>().Int32Value(); 
    int Cin = info[5].As<Napi::Number>().Int32Value(); 
    int H = info[6].As<Napi::Number>().Int32Value(); 
    int W = info[7].As<Napi::Number>().Int32Value(); 
    int Cout = info[8].As<Napi::Number>().Int32Value(); 
    int Kh = info[9].As<Napi::Number>().Int32Value(); 
    int Kw = info[10].As<Napi::Number>().Int32Value();

    float* input_data = input.Data();
    float* d = delta.Data();
    float* wg = weightGrads.Data();

    int padH = Kh / 2;
    int padW = Kw / 2;

    for (size_t f = 0; f < Cout; f++) {
        for (size_t kh = 0; kh < Kh; kh++) {
            for (size_t kw = 0; kw < Kw; kw++) {
                for (size_t c = 0; c < Cin; c++) {
                    float sum = 0.0f;
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            int inH = h + kh - padH;
                            int inW = w + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {

                                int inputIndex = (inH * inputW + inW) * Cin + c;

                                int deltaIndex = (h * W + w) * Cout + f;

                                sum += input_data[inputIndex] * d[deltaIndex];
                            }
                        }
                    }
                    int gradIndex = ((f * Kh + kh) * Kw + kw) * Cin + c;

                    wg[gradIndex] += sum;
                }
            }
        }
    }
    return weightGrads;
}

Napi::Value computeBiasGradsForConvWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    int outH = info[2].As<Napi::Number>().Int32Value();
    int outW = info[3].As<Napi::Number>().Int32Value();
    int numFilters = info[4].As<Napi::Number>().Int32Value();

    float* bg = biasGrads.Data();
    float* d = deltas.Data();

    for (size_t f = 0; f < numFilters; f++) {
        float sum = 0.0f;

        for (size_t h = 0; h < outH; h++) {
            for (size_t w = 0; w < outW; w++) {
                int idx = (h * outW + w) * numFilters + f;
                sum += d[idx];
            }
        }
        bg[f] += sum;
    }

    return biasGrads;
}



/* ================ module exports ===================*/
void GradientCalculationRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("computeWeightGradientsForWeightsInConnectedLayer", Napi::Function::New(env, ComputeGradientForDenseWeightsWrapper));
    exports.Set("computeKernelGradients", Napi::Function::New(env, computeKernelGradients));
    exports.Set("computeBiasGradsForConnected_Layer", Napi::Function::New(env, computeBiasGradsForConnected_LayerWrapper));
    exports.Set("computeBiasGradsForConv", Napi::Function::New(env, computeBiasGradsForConvWrapper));
}