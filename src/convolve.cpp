#include <napi.h>
#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;


// ======================= UTILS ================================ //
static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

// ==================== MAIN FUNCTIONS ======================= //
Napi::Value Convolve_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    int pointer = info[6].As<Napi::Number>().Int32Value();
    std::string modelID = info[7].As<Napi::String>().Utf8Value();
    std::string layerID = info[8].As<Napi::String>().Utf8Value();
    size_t inputSize = input.ElementLength();

    int numFilters = kernelShape[0];
    int kernelH = kernelShape[1];
    int kernelW = kernelShape[2];
    int depth = kernelShape[3];

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int inputDepth = inputShape[2];
    int outputH = outputShape[0];
    int outputW = outputShape[1];
    int outputDepth = outputShape[2];

    if (depth != inputDepth || numFilters != outputDepth) {
        throw std::invalid_argument("ConvolveForward input, output, and kernel depths do not match.");
    }

    const size_t expectedInputSize = static_cast<size_t>(inputH) * inputW * inputDepth;
    const size_t expectedKernelSize = static_cast<size_t>(numFilters) * kernelH * kernelW * depth;

    if (input.ElementLength() != expectedInputSize) {
        throw std::invalid_argument("ConvolveForward input length does not match inputShape.");
    }
    if (info[4].As<Napi::Float32Array>().ElementLength() != expectedKernelSize) {
        throw std::invalid_argument("ConvolveForward weight length does not match kernelShape.");
    }
    if (info[5].As<Napi::Float32Array>().ElementLength() != static_cast<size_t>(numFilters)) {
        throw std::invalid_argument("ConvolveForward bias length does not match the number of filters.");
    }

    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward", outputSize);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("convolve");

    cl_mem inputTensor = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(inputSize));
    clEnqueueWriteBuffer(queue, inputTensor, CL_FALSE, 0, sizeof(float) * inputSize, input.Data(), 0, nullptr, nullptr);
    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem biases = gpu.getBiases(modelID, pointer);
    cl_mem output_tensor = gpu.getOrCreate_Z(modelID, layerID, static_cast<size_t>(outputSize));


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_tensor);
    clSetKernelArg(kernel, 4, sizeof(int), &outputH);
    clSetKernelArg(kernel, 5, sizeof(int), &outputW);
    clSetKernelArg(kernel, 6, sizeof(int), &numFilters);
    clSetKernelArg(kernel, 7, sizeof(int), &kernelH);
    clSetKernelArg(kernel, 8, sizeof(int), &kernelW);
    clSetKernelArg(kernel, 9, sizeof(int), &depth);
    clSetKernelArg(kernel, 10, sizeof(int), &inputH);
    clSetKernelArg(kernel, 11, sizeof(int), &inputW);

    size_t global[3] = {
        (size_t)outputH,
        (size_t)outputW,
        (size_t)numFilters
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output_tensor, CL_TRUE, 0, sizeof(float) * outputSize, output.Data(), 0, nullptr, nullptr);

    return output;
}

Napi::Value Convolve_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[4].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[5].As<Napi::Float32Array>();
    int pointer = info[6].As<Napi::Number>().Int32Value();
    std::string modelID = info[7].As<Napi::String>().Utf8Value();
    std::string layerID = info[8].As<Napi::String>().Utf8Value();

    const int inputH = inputShape[0];
    const int inputW = inputShape[1];
    const int inputDepth = inputShape[2];
    const int outputH = outputShape[0];
    const int outputW = outputShape[1];
    const int outputDepth = outputShape[2];
    const int numFilters = kernelShape[0];
    const int kernelH = kernelShape[1];
    const int kernelW = kernelShape[2];
    const int depth = kernelShape[3];

    if (depth != inputDepth || numFilters != outputDepth) {
        throw std::invalid_argument("ConvolveForward input, output, and kernel depths do not match.");
    }
    if (inputTensor.ElementLength() != static_cast<size_t>(inputH) * inputW * inputDepth) {
        throw std::invalid_argument("ConvolveForward input length does not match inputShape.");
    }
    if (weightsArray.ElementLength() != static_cast<size_t>(numFilters) * kernelH * kernelW * depth) {
        throw std::invalid_argument("ConvolveForward weight length does not match kernelShape.");
    }
    if (biasesArray.ElementLength() != static_cast<size_t>(numFilters)) {
        throw std::invalid_argument("ConvolveForward bias length does not match the number of filters.");
    }

    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array outputTensor = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward", outputSize);

    const float* input = inputTensor.Data();
    const float* weights = weightsArray.Data();
    const float* biases = biasesArray.Data();
    float* output = outputTensor.Data();

    for (int h = 0; h < outputH; h++) {
        for (int w = 0; w < outputW; w++) {
            const size_t outputBase = (static_cast<size_t>(h) * outputW + w) * outputDepth;

            for (int filter = 0; filter < numFilters; filter++) {
                float sum = biases[filter];
                const size_t filterBase = static_cast<size_t>(filter) * kernelH * kernelW * depth;

                for (int kh = 0; kh < kernelH; kh++) {
                    const int inputHIndex = h + kh;
                    if (inputHIndex >= inputH) continue;

                    for (int kw = 0; kw < kernelW; kw++) {
                        const int inputWIndex = w + kw;
                        if (inputWIndex >= inputW) continue;

                        const size_t inputBase = (static_cast<size_t>(inputHIndex) * inputW + inputWIndex) * inputDepth;
                        const size_t weightBase = filterBase + (static_cast<size_t>(kh) * kernelW + kw) * depth;
                        int channel = 0;
                        for (; channel <= depth - 4; channel += 4) {
                            const float input0 = input[inputBase + channel];
                            const float input1 = input[inputBase + channel + 1];
                            const float input2 = input[inputBase + channel + 2];
                            const float input3 = input[inputBase + channel + 3];
                            const float weight0 = weights[weightBase + channel];
                            const float weight1 = weights[weightBase + channel + 1];
                            const float weight2 = weights[weightBase + channel + 2];
                            const float weight3 = weights[weightBase + channel + 3];
                            sum += input0 * weight0;
                            sum += input1 * weight1;
                            sum += input2 * weight2;
                            sum += input3 * weight3;
                        }
                        for (; channel < depth; channel++) {
                            sum += input[inputBase + channel] * weights[weightBase + channel];
                        }
                    }
                }
                output[outputBase + filter] = sum;
            }
        }
    }
    return outputTensor;
}

Napi::Value ConvolveDelta_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray outputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray deltaShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    int pointer = info[5].As<Napi::Number>().Int32Value();
    std::string modelID = info[6].As<Napi::String>().Utf8Value();
    std::string layerID = info[7].As<Napi::String>().Utf8Value();

    int F = kernelShape[0];
    int KH = kernelShape[1];
    int KW = kernelShape[2];
    int C_k = kernelShape[3];
    int deltaH = deltaShape[0];
    int deltaW = deltaShape[1];
    int deltaDepth = deltaShape[2];
    int oH = outputShape[0];
    int oW = outputShape[1];
    int outputDepth = outputShape[2];

    if (deltaDepth != F || outputDepth != C_k) {
        throw std::invalid_argument("ConvolveBackward delta, output, and kernel depths do not match.");
    }
    if (inputTensor.ElementLength() != static_cast<size_t>(deltaH) * deltaW * deltaDepth) {
        throw std::invalid_argument("ConvolveBackward input length does not match deltaShape.");
    }
    if (info[4].As<Napi::Float32Array>().ElementLength() != static_cast<size_t>(F) * KH * KW * C_k) {
        throw std::invalid_argument("ConvolveBackward weight length does not match kernelShape.");
    }

    int targetSize = oH * oW * outputDepth;
    const size_t deltaSize = static_cast<size_t>(deltaH) * deltaW * deltaDepth;
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_convolve_backward", targetSize);
    
    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("delta_convolve");

    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem delta = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_incoming_delta", deltaSize);
    clEnqueueWriteBuffer(queue, delta, CL_FALSE, 0, sizeof(float) * deltaSize, inputTensor.Data(), 0, nullptr, nullptr);
    cl_mem outputBuf = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_output_conv_backward", static_cast<size_t>(targetSize));


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &deltaH);
    clSetKernelArg(kernel, 4, sizeof(int), &deltaW);
    clSetKernelArg(kernel, 5, sizeof(int), &F);
    clSetKernelArg(kernel, 6, sizeof(int), &KH);
    clSetKernelArg(kernel, 7, sizeof(int), &KW);
    clSetKernelArg(kernel, 8, sizeof(int), &C_k);
    clSetKernelArg(kernel, 9, sizeof(int), &oH);
    clSetKernelArg(kernel, 10, sizeof(int), &oW);

    size_t global[3] = {
        (size_t)oH,
        (size_t)oW,
        (size_t)C_k
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, outputBuf, CL_TRUE, 0, sizeof(float) * targetSize, output.Data(), 0, nullptr, nullptr);
    
    return output;
}

Napi::Value ConvolveDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray outputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray deltaShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    Napi::Float32Array kernelsArray = info[4].As<Napi::Float32Array>();
    std::string modelID = info[6].As<Napi::String>().Utf8Value();
    std::string layerID = info[7].As<Napi::String>().Utf8Value();

    const int outputH = outputShape[0];
    const int outputW = outputShape[1];
    const int outputDepth = outputShape[2];
    const int deltaH = deltaShape[0];
    const int deltaW = deltaShape[1];
    const int deltaDepth = deltaShape[2];
    const int numFilters = kernelShape[0];
    const int kernelH = kernelShape[1];
    const int kernelW = kernelShape[2];
    const int depth = kernelShape[3];

    if (deltaDepth != numFilters || outputDepth != depth) {
        throw std::invalid_argument("ConvolveBackward delta, output, and kernel depths do not match.");
    }
    if (inputTensor.ElementLength() != static_cast<size_t>(deltaH) * deltaW * deltaDepth) {
        throw std::invalid_argument("ConvolveBackward input length does not match deltaShape.");
    }
    if (kernelsArray.ElementLength() != static_cast<size_t>(numFilters) * kernelH * kernelW * depth) {
        throw std::invalid_argument("ConvolveBackward weight length does not match kernelShape.");
    }

    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array outputTensor = getOrCreateBuffer(env, modelID, layerID+"_convolve_backward",  outputSize);
    const float* delta = inputTensor.Data();
    const float* weights = kernelsArray.Data();
    float* output = outputTensor.Data();

    for (int h = 0; h < outputH; h++) {
        for (int w = 0; w < outputW; w++) {
            const size_t outputBase = (static_cast<size_t>(h) * outputW + w) * outputDepth;

            for (int channel = 0; channel < outputDepth; channel++) {
                float sum = 0.0f;
                for (int kh = 0; kh < kernelH; kh++) {
                    const int deltaHIndex = h - kh;
                    if (deltaHIndex < 0 || deltaHIndex >= deltaH) continue;

                    for (int kw = 0; kw < kernelW; kw++) {
                        const int deltaWIndex = w - kw;
                        if (deltaWIndex < 0 || deltaWIndex >= deltaW) continue;

                        const size_t deltaBase = (static_cast<size_t>(deltaHIndex) * deltaW + deltaWIndex) * deltaDepth;
                        const size_t kernelBase = (static_cast<size_t>(kernelH - 1 - kh) * kernelW +
                            (kernelW - 1 - kw)) * depth + channel;

                        int filter = 0;
                        for (; filter <= numFilters - 4; filter += 4) {
                            const float delta0 = delta[deltaBase + filter];
                            const float delta1 = delta[deltaBase + filter + 1];
                            const float delta2 = delta[deltaBase + filter + 2];
                            const float delta3 = delta[deltaBase + filter + 3];
                            const size_t weight0 = (static_cast<size_t>(filter) * kernelH * kernelW * depth) + kernelBase;
                            const size_t weight1 = (static_cast<size_t>(filter + 1) * kernelH * kernelW * depth) + kernelBase;
                            const size_t weight2 = (static_cast<size_t>(filter + 2) * kernelH * kernelW * depth) + kernelBase;
                            const size_t weight3 = (static_cast<size_t>(filter + 3) * kernelH * kernelW * depth) + kernelBase;
                            sum += delta0 * weights[weight0];
                            sum += delta1 * weights[weight1];
                            sum += delta2 * weights[weight2];
                            sum += delta3 * weights[weight3];
                        }
                        for (; filter < numFilters; filter++) {
                            const size_t weight = (static_cast<size_t>(filter) * kernelH * kernelW * depth) + kernelBase;
                            sum += delta[deltaBase + filter] * weights[weight];
                        }
                    }
                }
                output[outputBase + channel] = sum;
            }
        }
    }
    return outputTensor;
}

/* ==================== Wrappers ======================== */
Napi::Value ConvolveWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return Convolve_GPU(info);
    }

    return Convolve_CPU(info);
}

Napi::Value ConvolveDeltaWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return ConvolveDelta_GPU(info);
    }

    return ConvolveDelta_CPU(info);
}

/* ==================== Module exports ======================== */
void ConvolveRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("ConvolveForward", Napi::Function::New(env, ConvolveWrapper));
    exports.Set("ConvolveBackward", Napi::Function::New(env, ConvolveDeltaWrapper));
}
