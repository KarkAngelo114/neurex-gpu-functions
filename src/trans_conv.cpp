#include <napi.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <vector>
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

static FloatArray Rotate_kernels(int F, int KH, int KW, int D, int pointer) {
    const auto& kernels_arr = getGlobalWeights(pointer);

    size_t kernel_length = kernels_arr.size();

    FloatArray outputData(kernel_length);

    const float* kernels = kernels_arr.data();
    float* rotated = outputData.data();

    for (size_t f = 0; f < F; f++) {
        for (size_t kh = 0; kh < KH; kh++) {
            for (size_t kw = 0; kw < KW; kw++) {
                for (size_t d = 0; d < D; d++) {
                    // Original Index
                    size_t oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    
                    // Rotated Index (Flip KH and KW)
                    size_t newKh = KH - 1 - kh;
                    size_t newKw = KW - 1 - kw;
                    size_t newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = kernels[oldIdx];
                }
            }
        }
    }

    return outputData;
}

Napi::Value TransConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_tensor = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    int outputH = info[2].As<Napi::Number>().Int32Value();
    int outputW = info[3].As<Napi::Number>().Int32Value();
    int num_filters = info[4].As<Napi::Number>().Int32Value();
    int kernel_height = info[5].As<Napi::Number>().Int32Value();
    int kernel_width = info[6].As<Napi::Number>().Int32Value();
    int depth = info[7].As<Napi::Number>().Int32Value();
    int inputH = info[8].As<Napi::Number>().Int32Value();
    int inputW = info[9].As<Napi::Number>().Int32Value();
    int pointer = info[10].As<Napi::Number>().Int32Value();
    int output_template_pointer = info[11].As<Napi::Number>().Int32Value();
    
    int outputSize = outputH * outputW * num_filters;
    Napi::Float32Array output_tensor = Napi::Float32Array::New(env, outputSize);
    FloatArray kernel_array = Rotate_kernels(num_filters, kernel_height, kernel_width, depth, pointer);
    FloatArray biases_array = getGlobalBiases(pointer);

    float* input = input_tensor.Data();
    float* kernel = kernel_array.data();
    float* biases = biases_array.data();
    float* output = output_tensor.Data();

    for (int f = 0; f < num_filters; f++) {
        float bias = biases[f];

        for (int y = 0; y < outputH; y++) {
            for (int x = 0; x < outputW; x++) {
                float sum = 0.0f;

                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        for (int c = 0; c < depth; c++) {
                            int inY = y * strides + ky;
                            int inX = x * strides + kx;

                            if (inY < inputH && inX < inputW) {
                                int inputIndex = ((inY * inputW + inX) * depth + c);
                                int kernelIndex = (((f * kernel_height + ky) * kernel_width + kx) * depth + c);
                                sum += input[inputIndex] * kernel[kernelIndex];
                            }
                        }
                    }
                }
                int outIndex = ((y * outputW + x) * num_filters + f);
                output[outIndex] = sum + bias;
            }
        }
    }

    return output_tensor;
    
}

Napi::Value TransConvDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input_tensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[2].As<Napi::Array>());
    int oH = info[3].As<Napi::Number>().Int32Value();
    int oW = info[4].As<Napi::Number>().Int32Value();
    int pointer = info[5].As<Napi::Number>().Int32Value();

    int Hp = inputShape[0];
    int Wp = inputShape[1];
    int C_in = inputShape[2];

    int F = kernelShape[0];
    int KH = kernelShape[1];
    int KW = kernelShape[2];
    int C_k = kernelShape[3];

    FloatArray kernels_array = getGlobalWeights(pointer);
    int outputSize = oH * oW * C_k;
    Napi::Float32Array output_tensor = Napi::Float32Array::New(env, outputSize);

    float* input = input_tensor.Data();
    float* kernel = kernels_array.data();
    float* output = output_tensor.Data();

    for (int c_out = 0; c_out < C_k; c_out++) {
        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                float sum = 0.0f;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int f = 0; f < F; f++) {
                            int ph = h + kh;
                            int pw = w + kw;
                            if (ph < Hp && pw < Wp) { 
                                int padIdx = (ph * Wp + pw) * C_in + f;
                                int kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
                                sum += input[padIdx] * kernel[kernelIdx];
                            }
                        }
                    }
                }
                output[(h * oW + w) * C_k + c_out] = sum;
            }
        }
    }
    return output_tensor;
}

Napi::Value TransConvWrapper(const Napi::CallbackInfo& info) {
    return TransConv_CPU(info);
}

Napi::Value TransConvDeltaWrapper(const Napi::CallbackInfo& info) {
    return TransConvDelta_CPU(info);
}

void TransConvFunc(const Napi::Env env, const Napi::Object exports) {
    exports.Set("TransConv", Napi::Function::New(env, TransConvWrapper));
    exports.Set("TransConvolveDelta", Napi::Function::New(env, TransConvDeltaWrapper));
}