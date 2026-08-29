#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "functions/functions.h"
#include <vector>
#include <cmath>
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


Napi::Value ComputeGradientForDenseWeights_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array activation_output = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputSize = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_kernel kernel = gpu.kernel("computeWeightGradsForConnected_Layer");
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem activations = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activation_output.ElementLength(), activation_output.Data(), nullptr);
    cl_mem deltaInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);
    cl_mem weight_grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &deltaInput);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_grads);
    clSetKernelArg(kernel, 3, sizeof(int), &inputSize);
    clSetKernelArg(kernel, 4, sizeof(int), &outputSize);

    size_t globalSize[2] = {
        (size_t)inputSize,
        (size_t)outputSize,
    };

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, weight_grads, CL_TRUE, 0, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), 0, nullptr, nullptr);

    clFinish(queue);

    clReleaseMemObject(activations);
    clReleaseMemObject(deltaInput);
    clReleaseMemObject(weight_grads);

    return weightGrads;

}

Napi::Value ComputeGradientForDenseWeights_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array activation_output = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputSize = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    float* a = activation_output.Data();
    float* d = deltas.Data();
    float* wg = weightGrads.Data();

    for (int i = 0; i < inputSize; i++) {
        float inputVal = a[i];
        int offset = i * outputSize;
        for (size_t j = 0; j < outputSize; j++) {
            wg[offset + j] += inputVal * d[j];
        }
    }
    return weightGrads;
}

Napi::Value computeBiasGradsForConnected_Layer_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasgrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    int biasGradsSize = biasgrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_kernel kernel = gpu.kernel("computeBiasGradsForConnected_Layer");
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem gradsInput = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasGradsSize, biasgrads.Data(), nullptr);
    cl_mem deltaInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &deltaInput);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradsInput);
    clSetKernelArg(kernel, 2, sizeof(int), &biasGradsSize);

    size_t globalSize = biasGradsSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // Read results back into the output array
    clEnqueueReadBuffer(queue, gradsInput, CL_TRUE, 0, sizeof(float) * biasGradsSize, biasgrads.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(gradsInput);
    clReleaseMemObject(deltaInput);

    return biasgrads;
}

Napi::Value computeBiasGradsForConnected_Layer_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasgrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();

    float* bg = biasgrads.Data();
    float* d = deltas.Data();
    size_t length = biasgrads.ElementLength();

    for (int i = 0; i < length; i++) {
        bg[i] += d[i];
    }

    return biasgrads;
}

Napi::Value computeKernelGradients_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltaTensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGradsTensor = info[2].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[3].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray kernelSize = Vectorize(info[5].As<Napi::Array>());
    int stride = info[6].As<Napi::Number>().Int32Value();

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];

    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];
    
    int Kh = kernelSize[0];
    int Kw = kernelSize[1];

    int padH = Kh / 2;
    int padW = Kw / 2;

    auto& gpu = GpuContext::instance();

    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("computeKernelGradients");

    cl_mem activations = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputTensor.ElementLength(), inputTensor.Data(), nullptr);
    cl_mem delta_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltaTensor.ElementLength(), deltaTensor.Data(), nullptr);
    cl_mem gradsArr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightGradsTensor.ElementLength(), weightGradsTensor.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradsArr);

    clSetKernelArg(kernel, 3, sizeof(int), &inputH);
    clSetKernelArg(kernel, 4, sizeof(int), &inputW);
    clSetKernelArg(kernel, 5, sizeof(int), &Cin);

    clSetKernelArg(kernel, 6, sizeof(int), &H);
    clSetKernelArg(kernel, 7, sizeof(int), &W);
    clSetKernelArg(kernel, 8, sizeof(int), &Cout);

    clSetKernelArg(kernel, 9, sizeof(int), &Kh);
    clSetKernelArg(kernel, 10, sizeof(int), &Kw);

    clSetKernelArg(kernel, 11, sizeof(int), &padH);
    clSetKernelArg(kernel, 12, sizeof(int), &padW);
    clSetKernelArg(kernel, 13, sizeof(int), &stride);

    // Calculate number of channel blocks (4 channels per block)
    int channelBlocks = (Cin + 3) / 4;

    size_t globalSize[3] = {
        (size_t)Cout,
        (size_t)Kh,
        (size_t)(Kw * channelBlocks)
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, gradsArr, CL_TRUE, 0, sizeof(float) * weightGradsTensor.ElementLength(), weightGradsTensor.Data(), 0, nullptr, nullptr);
 
    clReleaseMemObject(activations);
    clReleaseMemObject(delta_input);
    clReleaseMemObject(gradsArr);

    return weightGradsTensor;
}

Napi::Value computeKernelGradients_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltaTensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGradsTensor = info[2].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[3].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray kernelSize = Vectorize(info[5].As<Napi::Array>());
    int stride = info[6].As<Napi::Number>().Int32Value();

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];

    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];
    
    int Kh = kernelSize[0];
    int Kw = kernelSize[1];

    int padH = Kh / 2;
    int padW = Kw / 2;

    float* input = inputTensor.Data();
    float* delta = deltaTensor.Data();
    float* weightGrads = weightGradsTensor.Data();

    for (int f = 0; f < Cout; f++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                int kernelRowOffset = (f * Kh + kh) * Kw + kw;

                int c = 0;
                for (; c <= Cin - 4; c += 4) {
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            int inH = (h * stride) + kh - padH;
                            int inW = (w * stride) + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {
                                int baseInputIndex = (inH * inputW + inW) * Cin;
                                int deltaIndex = (h * W + w) * Cout + f;
                                float deltaVal = delta[deltaIndex];

                                sum0 += input[baseInputIndex + c] * deltaVal;
                                sum1 += input[baseInputIndex + c + 1] * deltaVal;
                                sum2 += input[baseInputIndex + c + 2] * deltaVal;
                                sum3 += input[baseInputIndex + c + 3] * deltaVal;
                            }
                        }
                    }

                    weightGrads[kernelRowOffset * Cin + c] += sum0;
                    weightGrads[kernelRowOffset * Cin + c + 1] += sum1;
                    weightGrads[kernelRowOffset * Cin + c + 2] += sum2;
                    weightGrads[kernelRowOffset * Cin + c + 3] += sum3;
                }

                // Process remaining channels
                for (; c < Cin; c++) {
                    float sum = 0.0f;

                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            int inH = (h * stride) + kh - padH;
                            int inW = (w * stride) + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {
                                int inputIndex = (inH * inputW + inW) * Cin + c;
                                int deltaIndex = (h * W + w) * Cout + f;
                                sum += input[inputIndex] * delta[deltaIndex];
                            }
                        }
                    }

                    int gradIndex = kernelRowOffset * Cin + c;
                    weightGrads[gradIndex] += sum;
                }
            }
        }
    }
    return weightGradsTensor;
}

Napi::Value computeBiasGradsForConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    int outH = info[2].As<Napi::Number>().Int32Value();
    int outW = info[3].As<Napi::Number>().Int32Value();
    int numFilters = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("computeBiasGradsForConv");
    
    cl_mem grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasGrads.ElementLength(), biasGrads.Data(), nullptr);
    cl_mem delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grads);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 2, sizeof(int), &outH);
    clSetKernelArg(kernel, 3, sizeof(int), &outW);
    clSetKernelArg(kernel, 4, sizeof(int), &numFilters);

    size_t globalSize = (size_t)numFilters;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(queue, grads, CL_TRUE, 0, sizeof(float) * biasGrads.ElementLength(), biasGrads.Data(), 0, nullptr, nullptr);
    clFinish(queue);
    clReleaseMemObject(grads);
    clReleaseMemObject(delta);

    return biasGrads;

}

Napi::Value computeBiasGradsForConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    size_t outH = info[2].As<Napi::Number>().Int32Value();
    size_t outW = info[3].As<Napi::Number>().Int32Value();
    int numFilters = info[4].As<Napi::Number>().Int32Value();

    float* bg = biasGrads.Data();
    float* d = deltas.Data();

    for (int f = 0; f < numFilters; f++) {
        float sum = 0.0f;

        for (size_t h = 0; h < outH; h++) {
            for (size_t w = 0; w < outW; w++) {
                size_t idx = (h * outW + w) * numFilters + f;
                sum += d[idx];
            }
        }
        bg[f] += sum;
    }

    return biasGrads;
}

Napi::Value recurrentWeightGradsAccumulation_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // 1. Extract Inputs & Raw Pointers directly
    float* activation_outputs = info[0].As<Napi::Float32Array>().Data();
    float* output = info[4].As<Napi::Float32Array>().Data();

    std::vector<const float*> hiddenStates = ExtractFloat32ArrayPointers(info[2].As<Napi::Array>());
    std::vector<const float*> deltaTs = ExtractFloat32ArrayPointers(info[3].As<Napi::Array>());

    IntArray weightShapeJS = Vectorize(info[5].As<Napi::Array>());
    int sequenceLength = info[6].As<Napi::Number>().Uint32Value();

    // 2. Setup Shape Parameters
    int featureSize = weightShapeJS[0];
    int units = weightShapeJS[1];
    size_t totalInputWeights = featureSize * units;

    // Helper buffer for h_prev when t == 0
    std::vector<float> zero_h_prev(units, 0.0f);

    // 3. Outer Product Loops (Pure C++)
    #pragma omp parallel for
    for (int t = 0; t < sequenceLength; ++t) {
        const float* x_t     = activation_outputs + (t * featureSize);
        const float* delta_t = deltaTs[t];
        
        // Handle t === 0 zero-fill fallback
        const float* h_prev = (t == 0 || hiddenStates[t - 1] == nullptr)  ? zero_h_prev.data() : hiddenStates[t - 1];

        // dL/dW_x += outer(x_t, delta_t)
        for (uint32_t i = 0; i < featureSize; ++i) {
            float xi = x_t[i];
            size_t rowOffset = i * units;
            for (uint32_t j = 0; j < units; ++j) {
                output[rowOffset + j] += xi * delta_t[j];
            }
        }

        // dL/dW_h += outer(h_prev, delta_t)
        for (uint32_t i = 0; i < units; ++i) {
            float hi = h_prev[i];
            size_t rowOffset = totalInputWeights + (i * units);
            for (uint32_t j = 0; j < units; ++j) {
                output[rowOffset + j] += hi * delta_t[j];
            }
        }
    }

    return info[4];
}

Napi::Value recurrentBiasGradsAccumulation_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads_array = info[0].As<Napi::Float32Array>();
    Napi::Array deltaTsJS = info[1].As<Napi::Array>();
    int sequenceLength = info[2].As<Napi::Number>().Int32Value();
    int units = info[3].As<Napi::Number>().Int32Value();

    std::vector<const float*> deltaTs = ExtractFloat32ArrayPointers(deltaTsJS);
    float* biasGrads = biasGrads_array.Data();

    for (int t = 0; t < sequenceLength; t++) {
        const float* delta_time_step = deltaTs[t];

        for (int j = 0; j < units; j++) {
            biasGrads[j] += delta_time_step[j];
        }
    }

    return biasGrads_array;
}

Napi::Value accumulateKernelGradsForTransConv_GPU(const Napi::CallbackInfo& info) {
     Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>(); // zeroed-template accumulator
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray inputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[6].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[7].As<Napi::Array>());

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
    int padTop = padH / 2;
    int padLeft = padW / 2;

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("accumulateTransConvKernelGrads");

    // Create GPU memory buffers
    cl_mem activations = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activation_outputs.ElementLength(), activation_outputs.Data(), nullptr);
    cl_mem delta_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);
    cl_mem grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grads);
    clSetKernelArg(kernel, 3, sizeof(int), &iH);
    clSetKernelArg(kernel, 4, sizeof(int), &iW);
    clSetKernelArg(kernel, 5, sizeof(int), &iD);
    clSetKernelArg(kernel, 6, sizeof(int), &oH);
    clSetKernelArg(kernel, 7, sizeof(int), &oW);
    clSetKernelArg(kernel, 8, sizeof(int), &f);
    clSetKernelArg(kernel, 9, sizeof(int), &kh);
    clSetKernelArg(kernel, 10, sizeof(int), &kw);
    clSetKernelArg(kernel, 11, sizeof(int), &strides);
    clSetKernelArg(kernel, 12, sizeof(int), &padTop);
    clSetKernelArg(kernel, 13, sizeof(int), &padLeft);

    // Enqueue kernel execution
    // Global size: (filters, kh, kw * iD)
    size_t globalSize[3] = {
        (size_t)f,
        (size_t)kh,
        (size_t)(kw * iD)
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, grads, CL_TRUE, 0, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(activations);
    clReleaseMemObject(delta_input);
    clReleaseMemObject(grads);

    return weightGrads;
}

Napi::Value accumulateKernelGradsForTransConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray inputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[6].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[7].As<Napi::Array>());

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
    int padTop = padH / 2;
    int padLeft = padW / 2;

    const float* activationData = activation_outputs.Data();
    const float* deltaData = deltas.Data();
    float* weightGradsData = weightGrads.Data();

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {
            int inputBase = (iy * iW + ix) * iD;

            for (int ky = 0; ky < kh; ky++) {
                int oy = iy * strides + ky - padTop;
                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {
                    int ox = ix * strides + kx - padLeft;
                    if (ox < 0 || ox >= oW) continue;

                    int deltaBase = (oy * oW + ox) * filters;

                    for (int filter = 0; filter < filters; filter++) {
                        float deltaVal = deltaData[deltaBase + filter];
                        int gradBase = ((filter * kh + ky) * kw + kx) * iD;

                        for (int c = 0; c < iD; c++) {
                            weightGradsData[gradBase + c] += activationData[inputBase + c] * deltaVal;
                        }
                    }
                }
            }
        }
    }

    return weightGrads;
}

// =================== wrapper ===================== //

Napi::Value computeBiasGradsForConnected_LayerWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeBiasGradsForConnected_Layer_GPU(info);
    }

    return computeBiasGradsForConnected_Layer_CPU(info);
}

Napi::Value ComputeGradientForDenseWeightsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ComputeGradientForDenseWeights_GPU(info);
    }
    return ComputeGradientForDenseWeights_CPU(info);
}

Napi::Value computeKernelGradientsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeKernelGradients_GPU(info);
    }

    return computeKernelGradients_CPU(info);
}

Napi::Value computeBiasGradsForConvWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeBiasGradsForConv_GPU(info);
    }

    return computeBiasGradsForConv_CPU(info);
}

Napi::Value recurrentWeightGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentWeightGradsAccumulation_CPU(info);
}

Napi::Value recurrentBiasGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentBiasGradsAccumulation_CPU(info);
}

Napi::Value accumulateKernelGradsForTransConvWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return accumulateKernelGradsForTransConv_GPU(info);
    }
    return accumulateKernelGradsForTransConv_CPU(info);
}


/* ================ module exports ===================*/
void GradientCalculationRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("computeWeightGradientsForWeightsInConnectedLayer", Napi::Function::New(env, ComputeGradientForDenseWeightsWrapper));
    exports.Set("computeKernelGradients", Napi::Function::New(env, computeKernelGradientsWrapper));
    exports.Set("computeBiasGradsForConnected_Layer", Napi::Function::New(env, computeBiasGradsForConnected_LayerWrapper));
    exports.Set("computeBiasGradsForConv", Napi::Function::New(env, computeBiasGradsForConvWrapper));
    exports.Set("recurrentWeightGradsAccumulation", Napi::Function::New(env, recurrentWeightGradsAccumulationWrapper));
    exports.Set("recurrentBiasGradsAccumulation", Napi::Function::New(env, recurrentBiasGradsAccumulationWrapper));
    exports.Set("accumulateKernelGradsForTransConv", Napi::Function::New(env, accumulateKernelGradsForTransConvWrapper));
}