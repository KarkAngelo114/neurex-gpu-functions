#include <napi.h>
#include <CL/cl.h>
#include <omp.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "functions/functions.h"
#include <vector>
#include <cmath>
#include <stdexcept>
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

Napi::Value accumulateWeightsAndBiasGradsForConnectedLayer_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activations_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads_tensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads_tensor = info[3].As<Napi::Float32Array>();
    IntArray weightShape = Vectorize(info[4].As<Napi::Array>());
    std::string modelID = info[5].As<Napi::String>().Utf8Value();
    std::string layerID = info[6].As<Napi::String>().Utf8Value();

    int inputSize = weightShape[0];
    int outputSize = weightShape[1];

    int weightGrads_size = weightGrads_tensor.ElementLength();
    int biasGrads_size = biasGrads_tensor.ElementLength();


    auto& gpu = GpuContext::instance();
    cl_kernel kernel = gpu.kernel("accumulateWeightsAndBiasGradsDense");
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem activations = gpu.getInput(modelID, layerID);
    cl_mem deltaInput = gpu.getDelta(modelID, layerID);

    cl_mem weight_grads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_dense_weight_grads", static_cast<size_t>(weightGrads_size));
    clEnqueueWriteBuffer(queue, weight_grads, CL_FALSE, 0, sizeof(float) * weightGrads_size, weightGrads_tensor.Data(), 0, nullptr, nullptr);

    cl_mem bias_grads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_dense_bias_grads", static_cast<size_t>(biasGrads_size));
    clEnqueueWriteBuffer(queue, bias_grads, CL_FALSE, 0, sizeof(float) * biasGrads_size, biasGrads_tensor.Data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &deltaInput);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_grads);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias_grads);
    clSetKernelArg(kernel, 4, sizeof(int), &inputSize);
    clSetKernelArg(kernel, 5, sizeof(int), &outputSize);

    size_t globalSize[2] = {
        (size_t)inputSize,
        (size_t)outputSize,
    };

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, weight_grads, CL_FALSE, 0, sizeof(float) * weightGrads_size, weightGrads_tensor.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, bias_grads, CL_FALSE, 0, sizeof(float) * biasGrads_size, biasGrads_tensor.Data(), 0, nullptr, nullptr);
    clFinish(queue);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads_tensor);
    output.Set("biasGrads", biasGrads_tensor);

    return output;

}

Napi::Value accumulateWeightsAndBiasGradsForConnectedLayer_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Napi::Float32Array activations_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads_tensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads_tensor = info[3].As<Napi::Float32Array>();
    IntArray weightShape = Vectorize(info[4].As<Napi::Array>());

    int inputSize = weightShape[0];
    int outputSize = weightShape[1];

    float* activations = activations_outputs.Data();
    float* delta = deltas.Data();
    float* weightGrads = weightGrads_tensor.Data();
    float* biasGrads = biasGrads_tensor.Data();

    // accumulate biasGrads
    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < deltas.ElementLength(); i++) {
        biasGrads[i] += delta[i];
    }

    // accumulate weightGrads
    for (int i = 0; i < inputSize; i++) {
        float inputVal = activations[i];
        int rowStart = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            weightGrads[rowStart + j] += inputVal * delta[j];
        }
    }


    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads_tensor);
    output.Set("biasGrads", biasGrads_tensor);

    return output;

}

Napi::Value AccumulateWeightAndBiasGradsForConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltaTensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGradsTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGradsTensor = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[6].As<Napi::Array>());
    int stride = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];

    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];
    
    int numFilters = kernelShape[0];
    int Kh = kernelShape[1];
    int Kw = kernelShape[2];

    int padH = Kh / 2;
    int padW = Kw / 2;

    size_t weightGradsSize = weightGradsTensor.ElementLength();
    size_t biasGradsSize = biasGradsTensor.ElementLength();

    auto& gpu = GpuContext::instance();

    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulateConvWeightandBiasGrads");

    cl_mem activations = gpu.getInput(modelID, layerID);
    cl_mem delta_input = gpu.getDelta(modelID, layerID);
    cl_mem weightGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_weight_grads", weightGradsSize);
    cl_mem biasGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_bias_grads", biasGradsSize);

    auto checkOpenCLError = [](cl_int error, const char* operation) {
        if (error != CL_SUCCESS) {
            throw std::runtime_error(std::string(operation) +
                " failed with OpenCL error " + std::to_string(error));
        }
    };

    auto setKernelArg = [&](cl_uint index, size_t size, const void* value) {
        checkOpenCLError(clSetKernelArg(kernel, index, size, value), "clSetKernelArg");
    };

    checkOpenCLError(clEnqueueWriteBuffer(queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGradsTensor.Data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(convolution weight gradients)");
    checkOpenCLError(clEnqueueWriteBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGradsTensor.Data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(convolution bias gradients)");

    setKernelArg(0, sizeof(cl_mem), &activations);
    setKernelArg(1, sizeof(cl_mem), &delta_input);
    setKernelArg(2, sizeof(cl_mem), &weightGradsBuffer);
    setKernelArg(3, sizeof(cl_mem), &biasGradsBuffer);
    setKernelArg(4, sizeof(int), &inputH);
    setKernelArg(5, sizeof(int), &inputW);
    setKernelArg(6, sizeof(int), &Cin);
    setKernelArg(7, sizeof(int), &H);
    setKernelArg(8, sizeof(int), &W);
    setKernelArg(9, sizeof(int), &Cout);
    setKernelArg(10, sizeof(int), &numFilters);
    setKernelArg(11, sizeof(int), &Kh);
    setKernelArg(12, sizeof(int), &Kw);
    setKernelArg(13, sizeof(int), &padH);
    setKernelArg(14, sizeof(int), &padW);
    setKernelArg(15, sizeof(int), &stride);

    size_t globalSize = static_cast<size_t>(numFilters);
    checkOpenCLError(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr),"clEnqueueNDRangeKernel(convolution gradient accumulation)");

    checkOpenCLError(clEnqueueReadBuffer(queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGradsTensor.Data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(convolution weight gradients)");
    checkOpenCLError(clEnqueueReadBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGradsTensor.Data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(convolution bias gradients)");
    clFinish(queue);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGradsTensor);
    output.Set("biasGrads", biasGradsTensor);
    return output;
}

Napi::Value AccumulateWeightAndBiasGradsForConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltaTensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGradsTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGradsTensor = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[6].As<Napi::Array>());
    int stride = info[7].As<Napi::Number>().Int32Value();

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];

    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];
    
    int numFilters = kernelShape[0];
    int Kh = kernelShape[1];
    int Kw = kernelShape[2];
    int d = kernelShape[3];

    int padH = Kh / 2;
    int padW = Kw / 2;

    float* input = inputTensor.Data();
    float* delta = deltaTensor.Data();
    float* weightGrads = weightGradsTensor.Data();
    float* biasGrads = biasGradsTensor.Data();

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

    for (int f = 0; f < numFilters; f++) {
        float sum = 0.0f;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (h * W + w) * numFilters + f;
                sum += delta[idx];
            }
        }

        biasGrads[f] += sum;
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGradsTensor);
    output.Set("biasGrads", biasGradsTensor);

    return output;
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

            #pragma omp unroll partial(4)
            for (uint32_t j = 0; j < units; ++j) {
                output[rowOffset + j] += xi * delta_t[j];
            }
        }

        // dL/dW_h += outer(h_prev, delta_t)
        for (uint32_t i = 0; i < units; ++i) {
            float hi = h_prev[i];
            size_t rowOffset = totalInputWeights + (i * units);

            #pragma omp unroll partial(4)
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

        #pragma omp unroll partial(4)
        for (int j = 0; j < units; j++) {
            biasGrads[j] += delta_time_step[j];
        }
    }

    return biasGrads_array;
}

Napi::Value accumulateWeightandBiasGradsForTransConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[6].As<Napi::Array>());
    int strides = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int filters = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = padH / 2;
    int padLeft = padW / 2;
    size_t weightGradsSize = weightGrads.ElementLength();
    size_t biasGradsSize = biasGrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulateTransConvWeightAndBiasGrads");

    cl_mem activations = gpu.getInput(modelID, layerID);
    cl_mem delta_input = gpu.getDelta(modelID, layerID);
    cl_mem weightGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_trans_conv_weight_grads", weightGradsSize);
    cl_mem biasGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_trans_conv_bias_grads", biasGradsSize);

    auto checkOpenCLError = [](cl_int error, const char* operation) {
        if (error != CL_SUCCESS) {
            throw std::runtime_error(std::string(operation) +
                " failed with OpenCL error " + std::to_string(error));
        }
    };

    auto setKernelArg = [&](cl_uint index, size_t size, const void* value) {
        checkOpenCLError(clSetKernelArg(kernel, index, size, value), "clSetKernelArg(transposed convolution gradients)");
    };

    checkOpenCLError(clEnqueueWriteBuffer(queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads.Data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(transposed convolution weight gradients)");
    checkOpenCLError(clEnqueueWriteBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGrads.Data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(transposed convolution bias gradients)");

    setKernelArg(0, sizeof(cl_mem), &activations);
    setKernelArg(1, sizeof(cl_mem), &delta_input);
    setKernelArg(2, sizeof(cl_mem), &weightGradsBuffer);
    setKernelArg(3, sizeof(cl_mem), &biasGradsBuffer);
    setKernelArg(4, sizeof(int), &iH);
    setKernelArg(5, sizeof(int), &iW);
    setKernelArg(6, sizeof(int), &iD);
    setKernelArg(7, sizeof(int), &oH);
    setKernelArg(8, sizeof(int), &oW);
    setKernelArg(9, sizeof(int), &filters);
    setKernelArg(10, sizeof(int), &kh);
    setKernelArg(11, sizeof(int), &kw);
    setKernelArg(12, sizeof(int), &strides);
    setKernelArg(13, sizeof(int), &padTop);
    setKernelArg(14, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        static_cast<size_t>(filters),
        static_cast<size_t>(kh),
        static_cast<size_t>(kw) * static_cast<size_t>(iD)
    };

    checkOpenCLError(clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel(transposed convolution gradients)");

    checkOpenCLError(clEnqueueReadBuffer( queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads.Data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(transposed convolution weight gradients)");
    checkOpenCLError(clEnqueueReadBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGrads.Data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(transposed convolution bias gradients)");
    checkOpenCLError(clFinish(queue), "clFinish(transposed convolution gradients)");

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;
}

Napi::Value accumulateWeightandBiasGradsForTransConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[6].As<Napi::Array>());
    int strides = info[7].As<Napi::Number>().Int32Value();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int filters = weightShape[0];
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
    float* biasGradsData = biasGrads.Data();

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

                        #pragma omp unroll partial(4)
                        for (int c = 0; c < iD; c++) {
                            weightGradsData[gradBase + c] += activationData[inputBase + c] * deltaVal;
                        }
                    }
                }
            }
        }
    }

    for (int f = 0; f < filters; f++) {
        float sum = 0.0f;

        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                int idx = (h * oW + w) * filters + f;
                sum += deltas[idx];
            }
        }

        biasGradsData[f] += sum;
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;

}

Napi::Value AccumulateAttentionWeightsGradients_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto dQ = info[0].As<Napi::Float32Array>();
    auto dK = info[1].As<Napi::Float32Array>();
    auto dV = info[2].As<Napi::Float32Array>();
    auto dMhaOutput = info[3].As<Napi::Float32Array>();
    auto mhaOutput = info[4].As<Napi::Float32Array>();
    auto activationOutputs = info[5].As<Napi::Float32Array>();
    auto weightGrads = info[6].As<Napi::Float32Array>();
    int embedDim = info[7].As<Napi::Number>().Int32Value();
    int seqLen = info[8].As<Napi::Number>().Int32Value();

    const float* deltas[] = {dQ.Data(), dK.Data(), dV.Data(), dMhaOutput.Data()};
    const float* inputs[] = {activationOutputs.Data(), activationOutputs.Data(), activationOutputs.Data(), mhaOutput.Data()};
    float* output = weightGrads.Data();
    size_t blockSize = static_cast<size_t>(embedDim) * embedDim;

    for (int block = 0; block < 4; block++) {
        for (int i = 0; i < embedDim; i++) {
            for (int j = 0; j < embedDim; j++) {
                float sum = 0.0f;

                int t = 0;
                for (; t + 3 < seqLen; t += 4) {
                    sum += inputs[block][t * embedDim + i] * deltas[block][t * embedDim + j];
                    sum += inputs[block][(t + 1) * embedDim + i] * deltas[block][(t + 1) * embedDim + j];
                    sum += inputs[block][(t + 2) * embedDim + i] * deltas[block][(t + 2) * embedDim + j];
                    sum += inputs[block][(t + 3) * embedDim + i] * deltas[block][(t + 3) * embedDim + j];
                }
                for (; t < seqLen; t++) {
                    sum += inputs[block][t * embedDim + i] * deltas[block][t * embedDim + j];
                }

                output[block * blockSize + i * embedDim + j] += sum;
            }
        }
    }

    return weightGrads;
}

Napi::Value AccumulateAttentionWeightsGradients_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto dQ = info[0].As<Napi::Float32Array>();
    auto dK = info[1].As<Napi::Float32Array>();
    auto dV = info[2].As<Napi::Float32Array>();
    auto dMhaOutput = info[3].As<Napi::Float32Array>();
    auto mhaOutput = info[4].As<Napi::Float32Array>();
    auto activationOutputs = info[5].As<Napi::Float32Array>();
    auto weightGrads = info[6].As<Napi::Float32Array>();
    int embedDim = info[7].As<Napi::Number>().Int32Value();
    int seqLen = info[8].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulate_attention_weight_grads");

    size_t tensorBytes = sizeof(float) * static_cast<size_t>(seqLen) * embedDim;
    size_t gradientBytes = sizeof(float) * weightGrads.ElementLength();

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, activationOutputs.Data(), nullptr);
    cl_mem mha = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, mhaOutput.Data(), nullptr);
    cl_mem q = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dQ.Data(), nullptr);
    cl_mem k = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dK.Data(), nullptr);
    cl_mem v = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dV.Data(), nullptr);
    cl_mem o = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dMhaOutput.Data(), nullptr);
    cl_mem grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gradientBytes, weightGrads.Data(), nullptr);

    cl_mem args[] = {input, mha, q, k, v, o, grads};

    for (int i = 0; i < 7; i++) {
        clSetKernelArg(kernel, i, sizeof(cl_mem), &args[i]);
    };

    clSetKernelArg(kernel, 7, sizeof(int), &embedDim);
    clSetKernelArg(kernel, 8, sizeof(int), &seqLen);

    size_t global[3] = {4, static_cast<size_t>(embedDim), static_cast<size_t>(embedDim)};

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, grads, CL_TRUE, 0, gradientBytes, weightGrads.Data(), 0, nullptr, nullptr);

    for (cl_mem buffer : args) {
        clReleaseMemObject(buffer);
    };

    return weightGrads;
}

Napi::Value AccumulateAttentionBiasGrads_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto dQ = info[0].As<Napi::Float32Array>();
    auto dK = info[1].As<Napi::Float32Array>();
    auto dV = info[2].As<Napi::Float32Array>();
    auto dMhaOutput = info[3].As<Napi::Float32Array>();
    auto biasGrads = info[4].As<Napi::Float32Array>();
    int embedDim = info[5].As<Napi::Number>().Int32Value();
    int seqLen = info[6].As<Napi::Number>().Int32Value();

    const float* deltas[] = {dQ.Data(), dK.Data(), dV.Data(), dMhaOutput.Data()};
    float* output = biasGrads.Data();
    
    for (int block = 0; block < 4; block++) {
        for (int j = 0; j < embedDim; j++) {

            int t = 0;
            for (; t + 3 < seqLen; t += 4) {
                output[block * embedDim + j] += deltas[block][t * embedDim + j];
                output[block * embedDim + j] += deltas[block][(t + 1) * embedDim + j];
                output[block * embedDim + j] += deltas[block][(t + 2) * embedDim + j];
                output[block * embedDim + j] += deltas[block][(t + 3) * embedDim + j];
            }
            for (; t < seqLen; t++) {
                output[block * embedDim + j] += deltas[block][t * embedDim + j];
            }
        }
    }

    return biasGrads;
}

Napi::Value AccumulateAttentionBiasGrads_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto dQ = info[0].As<Napi::Float32Array>();
    auto dK = info[1].As<Napi::Float32Array>();
    auto dV = info[2].As<Napi::Float32Array>();
    auto dMhaOutput = info[3].As<Napi::Float32Array>();
    auto biasGrads = info[4].As<Napi::Float32Array>();
    int embedDim = info[5].As<Napi::Number>().Int32Value();
    int seqLen = info[6].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulate_attention_bias_grads");

    size_t tensorBytes = sizeof(float) * static_cast<size_t>(seqLen) * embedDim;
    size_t gradientBytes = sizeof(float) * biasGrads.ElementLength();

    cl_mem q = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dQ.Data(), nullptr);
    cl_mem k = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dK.Data(), nullptr);
    cl_mem v = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dV.Data(), nullptr);
    cl_mem o = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorBytes, dMhaOutput.Data(), nullptr);
    cl_mem grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gradientBytes, biasGrads.Data(), nullptr);

    cl_mem args[] = {q, k, v, o, grads};
    
    for (int i = 0; i < 5; i++) {
        clSetKernelArg(kernel, i, sizeof(cl_mem), &args[i]);
    };

    clSetKernelArg(kernel, 5, sizeof(int), &embedDim);
    clSetKernelArg(kernel, 6, sizeof(int), &seqLen);

    size_t global[2] = {4, static_cast<size_t>(embedDim)};

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, grads, CL_TRUE, 0, gradientBytes, biasGrads.Data(), 0, nullptr, nullptr);

    for (cl_mem buffer : args) {
        clReleaseMemObject(buffer);
    };
    
    return biasGrads;
}

Napi::Value AccumulateGammaAndBetaGrads_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array gammaGrads_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array dGamma_array = info[1].As<Napi::Float32Array>();
    Napi::Float32Array betaGrads_array = info[2].As<Napi::Float32Array>();
    Napi::Float32Array dBeta_array = info[3].As<Napi::Float32Array>();
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();
    int size = gammaGrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulate_gamma_beta_grads");

    cl_mem dGamma = gpu.get_dGamma(modelID, layerID);
    cl_mem dBeta = gpu.get_dBeta(modelID, layerID);
    cl_mem gammaGrads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_gamma_grads_input", static_cast<size_t>(size));
    cl_mem betaGrads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_beta_grads_input", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, gammaGrads, CL_FALSE, 0, sizeof(float) * size, gammaGrads_array.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, betaGrads, CL_FALSE, 0, sizeof(float) * size, betaGrads_array.Data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dGamma);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dBeta);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gammaGrads);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &betaGrads);
    clSetKernelArg(kernel, 4, sizeof(int), &size);

    size_t globalSize = (size_t)size;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, gammaGrads, CL_FALSE, 0, sizeof(float) * size, gammaGrads_array.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, betaGrads, CL_FALSE, 0, sizeof(float) * size, betaGrads_array.Data(), 0, nullptr, nullptr);
    clFinish(queue);

    Napi::Object output = Napi::Object::New(env);
    output.Set("gammaGrads", gammaGrads_array);
    output.Set("betaGrads", betaGrads_array);
}

Napi::Value AccumulateGammaAndBetaGrads_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array gammaGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array dGamma = info[1].As<Napi::Float32Array>();
    Napi::Float32Array betaGrads = info[2].As<Napi::Float32Array>();
    Napi::Float32Array dBeta = info[3].As<Napi::Float32Array>();

    float* g = gammaGrads.Data();
    float* dG = dGamma.Data();
    float* b = betaGrads.Data();
    float* dB = dGamma.Data();

    int length = grads.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        g[i] += dG[i];
        b[i] += dB[i];
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("gammaGrads", gammaGrads);
    output.Set("betaGrads", betaGrads);

    return output;
}

// =================== wrappers ===================== //

Napi::Value accumulateWeightsAndBiasGradsForConnectedLayer(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return accumulateWeightsAndBiasGradsForConnectedLayer_GPU(info);
    }

    return accumulateWeightsAndBiasGradsForConnectedLayer_CPU(info);
}

Napi::Value AccumulateWeightAndBiasGradsForConv(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return AccumulateWeightAndBiasGradsForConv_GPU(info);
    }

    return AccumulateWeightAndBiasGradsForConv_CPU(info);
}

Napi::Value recurrentWeightGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentWeightGradsAccumulation_CPU(info);
}

Napi::Value recurrentBiasGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentBiasGradsAccumulation_CPU(info);
}

Napi::Value accumulateWeightandBiasGradsForTransConv_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return accumulateWeightandBiasGradsForTransConv_GPU(info);
    }
    
    return accumulateWeightandBiasGradsForTransConv_CPU(info);
}

Napi::Value AccumulateAttentionWeightsGradients_Wrapper(const Napi::CallbackInfo& info) {
    // if (get_Global_Boolean_On_GPU()) return AccumulateAttentionWeightsGradients_GPU(info);
    return AccumulateAttentionWeightsGradients_CPU(info);
}

Napi::Value AccumulateAttentionBiasGrads_Wrapper(const Napi::CallbackInfo& info) {
    // if (getComputeBackendType() == "opencl") return AccumulateAttentionBiasGrads_GPU(info);
    return AccumulateAttentionBiasGrads_CPU(info);
}

Napi::Value AccumulateGammaAndBetaGrads_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return AccumulateGammaAndBetaGrads_GPU(info);
    }
    return AccumulateGammaAndBetaGrads_CPU(info);
}

/* ================ module exports ===================*/
void GradientCalculationRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("accumulateWeightsAndBiasGradsForConnectedLayer", Napi::Function::New(env, accumulateWeightsAndBiasGradsForConnectedLayer));
    exports.Set("AccumulateWeightAndBiasGradsForConv", Napi::Function::New(env, AccumulateWeightAndBiasGradsForConv));
    exports.Set("computeBiasGradsForConv", Napi::Function::New(env, computeBiasGradsForConvWrapper));
    exports.Set("recurrentWeightGradsAccumulation", Napi::Function::New(env, recurrentWeightGradsAccumulationWrapper));
    exports.Set("recurrentBiasGradsAccumulation", Napi::Function::New(env, recurrentBiasGradsAccumulationWrapper));
    exports.Set("accumulateWeightandBiasGradsForTransConv", Napi::Function::New(env, accumulateWeightandBiasGradsForTransConv_wrapper));
    exports.Set("accumulateAttentionWeightsGradients", Napi::Function::New(env, AccumulateAttentionWeightsGradients_Wrapper));
    exports.Set("accumulateAttentionBiasGrads", Napi::Function::New(env, AccumulateAttentionBiasGrads_Wrapper));
    exports.Set("AccumulateGammaAndBetaGrads", Napi::Function::New(env, AccumulateGammaAndBetaGrads_wrapper));
}