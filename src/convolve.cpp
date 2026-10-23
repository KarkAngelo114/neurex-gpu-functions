#include <napi.h>
#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "global_activations/global_activations.h"
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

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    std::string activationType = info[4].As<Napi::String>().Utf8Value();
    int pointer = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();
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


    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_activated_output_vector", outputSize);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel convolveKernel = gpu.kernel("convolve");
    cl_kernel activationKernel = gpu.kernel(activationType+"_op");

    cl_mem inputTensor = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(inputSize));
    clEnqueueWriteBuffer(queue, inputTensor, CL_FALSE, 0, sizeof(float) * inputSize, input.Data(), 0, nullptr, nullptr);
    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem biases = gpu.getBiases(modelID, pointer);
    cl_mem z_values = gpu.getOrCreate_Z(modelID, layerID, static_cast<size_t>(outputSize));
    cl_mem activated_output = gpu.getOrCreate_ActivationOutput(modelID, layerID, static_cast<size_t>(outputSize));


    clSetKernelArg(convolveKernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(convolveKernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(convolveKernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(convolveKernel, 3, sizeof(cl_mem), &z_values);
    clSetKernelArg(convolveKernel, 4, sizeof(int), &outputH);
    clSetKernelArg(convolveKernel, 5, sizeof(int), &outputW);
    clSetKernelArg(convolveKernel, 6, sizeof(int), &numFilters);
    clSetKernelArg(convolveKernel, 7, sizeof(int), &kernelH);
    clSetKernelArg(convolveKernel, 8, sizeof(int), &kernelW);
    clSetKernelArg(convolveKernel, 9, sizeof(int), &depth);
    clSetKernelArg(convolveKernel, 10, sizeof(int), &inputH);
    clSetKernelArg(convolveKernel, 11, sizeof(int), &inputW);

    size_t global[3] = {
        (size_t)outputH,
        (size_t)outputW,
        (size_t)numFilters
    };

    clEnqueueNDRangeKernel(queue, convolveKernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, z_values, CL_TRUE, 0, sizeof(float) * outputSize, z_values_vector.Data(), 0, nullptr, nullptr);

    if (activationType != "softmax") {
        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &activated_output);
        clSetKernelArg(activationKernel, 2, sizeof(int), &outputSize);

        // run activation function
        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        
        clEnqueueReadBuffer(queue, activated_output, CL_TRUE, 0, sizeof(float) * outputSize, activated_output_vector.Data(), 0, nullptr, nullptr);
    }
    else {
        float* data = z_values_vector.Data();

        float max_val = data[0];
        for (int i = 1; i < outputSize; i++) {
            max_val = std::max(max_val, data[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            sum += std::exp(data[i] - max_val);
        }

        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &activated_output);
        clSetKernelArg(activationKernel, 2, sizeof(float), &max_val);
        clSetKernelArg(activationKernel, 3, sizeof(float), &sum);
        clSetKernelArg(activationKernel, 4, sizeof(int), &outputSize);

        size_t softmax_branch_global_work_size = (size_t)outputSize;

        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &softmax_branch_global_work_size, nullptr, 0, nullptr, nullptr);

        clEnqueueReadBuffer(queue, activated_output, CL_TRUE, 0, sizeof(float) * outputSize, activated_output_vector.Data(), 0, nullptr, nullptr);
    }

    Napi::Object output_obj = Napi::Object::New(env);
    output_obj.Set("z_values", z_values_vector);
    output_obj.Set("activated_output",activated_output_vector);

    return output_obj;
}

Napi::Value Convolve_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    std::string activationType = info[4].As<Napi::String>().Utf8Value();
    Napi::Float32Array weightsArray = info[5].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[6].As<Napi::Float32Array>();
    int pointer = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

    std::unordered_map<std::string, void(*)(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn)> activations = {
        {"linear", LINEAR},
        {"relu", RELU},
        {"sigmoid", SIGMOID},
        {"tanh", TANH},
        {"softmax", SOFTMAX}
    };

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

    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_activated_output", outputSize);

    const float* input = inputTensor.Data();
    const float* weights = weightsArray.Data();
    const float* biases = biasesArray.Data();
    float* output = z_values_vector.Data();
    const int padTop = outputH == inputH ? (kernelH - 1) / 2 : 0;
    const int padLeft = outputW == inputW ? (kernelW - 1) / 2 : 0;

    for (int h = 0; h < outputH; h++) {
        for (int w = 0; w < outputW; w++) {
            const size_t outputBase = (static_cast<size_t>(h) * outputW + w) * outputDepth;

            for (int filter = 0; filter < numFilters; filter++) {
                float sum = biases[filter];
                const size_t filterBase = static_cast<size_t>(filter) * kernelH * kernelW * depth;

                for (int kh = 0; kh < kernelH; kh++) {
                    const int inputHIndex = h + kh - padTop;
                    if (inputHIndex < 0 || inputHIndex >= inputH) continue;

                    for (int kw = 0; kw < kernelW; kw++) {
                        const int inputWIndex = w + kw - padLeft;
                        if (inputWIndex < 0 || inputWIndex >= inputW) continue;

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

    activations[activationType](z_values_vector, activated_output_vector);

    Napi::Object output_obj = Napi::Object::New(env);
    output_obj.Set("z_values", z_values_vector);
    output_obj.Set("activated_output",activated_output_vector);

    return output_obj;
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
    const int padTop = deltaH == outputH ? (kernelH - 1) / 2 : 0;
    const int padLeft = deltaW == outputW ? (kernelW - 1) / 2 : 0;

    std::fill(output, output + outputSize, 0.0f);

    for (int h = 0; h < deltaH; h++) {
        for (int w = 0; w < deltaW; w++) {
            const size_t deltaBase = (static_cast<size_t>(h) * deltaW + w) * deltaDepth;

            for (int filter = 0; filter < numFilters; filter++) {
                const float deltaValue = delta[deltaBase + filter];

                for (int kh = 0; kh < kernelH; kh++) {
                    const int inputHIndex = h + kh - padTop;
                    if (inputHIndex < 0 || inputHIndex >= outputH) continue;

                    for (int kw = 0; kw < kernelW; kw++) {
                        const int inputWIndex = w + kw - padLeft;
                        if (inputWIndex < 0 || inputWIndex >= outputW) continue;

                        const size_t outputBase = (static_cast<size_t>(inputHIndex) * outputW + inputWIndex) * outputDepth;
                        const size_t weightBase = (static_cast<size_t>(filter) * kernelH * kernelW +
                            static_cast<size_t>(kh) * kernelW + kw) * depth;

                        for (int channel = 0; channel < depth; channel++) {
                            output[outputBase + channel] += deltaValue * weights[weightBase + channel];
                        }
                    }
                }
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
