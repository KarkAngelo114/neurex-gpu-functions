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

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    std::string activationType = info[4].As<Napi::String>().Utf8Value();
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


    int outputSize = outputH * outputW * outputDepth;
    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_convolve_forward_activated_output_vector", outputSize);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel convolveKernel = gpu.kernel("convolve");
    cl_kernel activationKernel = gpu.kernel(activationType+"_op");

    cl_mem inputTensor = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(inputSize));
    clEnqueueWriteBuffer(queue, inputTensor, CL_FALSE, 0, sizeof(float) * inputSize, input.Data(), 0, nullptr, nullptr);
    cl_mem weights = gpu.getWeights(modelID, layerID);
    cl_mem biases = gpu.getBiases(modelID, layerID);
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

    size_t activationSize = (size_t)outputSize;

    clEnqueueNDRangeKernel(queue, convolveKernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, z_values, CL_TRUE, 0, sizeof(float) * outputSize, z_values_vector.Data(), 0, nullptr, nullptr);

    if (activationType != "softmax") {
        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &activated_output);
        clSetKernelArg(activationKernel, 2, sizeof(int), &outputSize);

        // run activation function
        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &activationSize, nullptr, 0, nullptr, nullptr);
        
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
    std::string modelID = info[7].As<Napi::String>().Utf8Value();
    std::string layerID = info[8].As<Napi::String>().Utf8Value();

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
    std::string modelID = info[5].As<Napi::String>().Utf8Value();
    std::string layerID = info[6].As<Napi::String>().Utf8Value();

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

    cl_mem weights = gpu.getWeights(modelID, layerID);
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

Napi::Value fusedFinalDeltaProjector_ForConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array zs_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array layerInput = info[1].As<Napi::Float32Array>();
    Napi::Float32Array layerOutput = info[2].As<Napi::Float32Array>();
    Napi::Float32Array incomingDelta = info[3].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads_array = info[4].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads_array = info[5].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[6].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[7].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[8].As<Napi::Array>());
    int stride = info[9].As<Napi::Number>().Int32Value();
    std::string activationType = info[10].As<Napi::String>().Utf8Value();
    bool applyDerivativeActivation = info[11].As<Napi::Boolean>().Value();
    std::string modelID = info[12].As<Napi::String>().Utf8Value();
    std::string layerID = info[13].As<Napi::String>().Utf8Value();

    if (inputShape.size() != 3 || outputShape.size() != 3 || kernelShape.size() != 4) {
        Napi::TypeError::New(env, "Convolution gradient shapes must have 3, 3, and 4 dimensions").ThrowAsJavaScriptException();
        return env.Null();
    }

    int size = incomingDelta.ElementLength();

    int numFilters = kernelShape[0];
    int Kh = kernelShape[1];
    int Kw = kernelShape[2];
    int d = kernelShape[3];
    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];
    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];

    if (inputH <= 0 || inputW <= 0 || Cin <= 0 || H <= 0 || W <= 0 ||
        numFilters <= 0 || Kh <= 0 || Kw <= 0 || stride <= 0 ||
        d != Cin || Cout != numFilters || size != H * W * Cout ||
        zs_array.ElementLength() != size || layerOutput.ElementLength() != size ||
        layerInput.ElementLength() != inputH * inputW * Cin ||
        weightGrads_array.ElementLength() != numFilters * Kh * Kw * Cin ||
        biasGrads_array.ElementLength() != numFilters) {
        Napi::RangeError::New(env, "Convolution gradient inputs do not match the provided shapes").ThrowAsJavaScriptException();
        return env.Null();
    }

    int padH = H == inputH ? (Kh - 1) / 2 : 0;
    int padW = W == inputW ? (Kw - 1) / 2 : 0;

    Napi::Float32Array finalDelta = getOrCreateBuffer(env, modelID, layerID + "_final_delta", size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel gradientAccumulationKernel = gpu.kernel("accumulateConvWeightandBiasGrads");

    cl_mem z_values = gpu.getZ(modelID, layerID);
    cl_mem layerInputBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_gradient_input", layerInput.ElementLength());
    cl_mem layerOutputBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_gradient_output", layerOutput.ElementLength());
    cl_mem finalDeltaBuffer = gpu.getOrCreate_Delta(modelID, layerID, static_cast<size_t>(size));
    cl_mem incoming_deltaBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_incoming_delta", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, layerInputBuffer, CL_FALSE, 0, sizeof(float) * layerInput.ElementLength(), layerInput.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, layerOutputBuffer, CL_FALSE, 0, sizeof(float) * layerOutput.ElementLength(), layerOutput.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, incoming_deltaBuffer, CL_FALSE, 0, sizeof(float) * size, incomingDelta.Data(), 0, nullptr, nullptr);
    cl_mem dAct = gpu.getOrCreate_DAct(modelID, layerID, static_cast<size_t>(size));

    size_t weightGradsSize = weightGrads_array.ElementLength();
    size_t biasGradsSize = biasGrads_array.ElementLength();
    cl_mem weightGrads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_weight_grads", weightGradsSize);
    cl_mem biasGrads = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_conv_bias_grads", biasGradsSize);
    clEnqueueWriteBuffer(queue, weightGrads, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads_array.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, biasGrads, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGrads_array.Data(), 0, nullptr, nullptr);

    size_t scalarSize = static_cast<size_t>(size);
    size_t gradientAccumulationSize = static_cast<size_t>(numFilters);

    if (applyDerivativeActivation && activationType == "softmax") {
        cl_kernel activationKernel = gpu.kernel("dsoftmax");
        const float* probabilities = layerOutput.Data();
        const float* delta = incomingDelta.Data();

        float dotProduct = 0.0f;
        #pragma omp unroll partial(4)
        for (int i = 0; i < size; i++) {
            dotProduct += probabilities[i] * delta[i];
        }

        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &layerOutputBuffer);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &incoming_deltaBuffer);
        clSetKernelArg(activationKernel, 2, sizeof(cl_mem), &finalDeltaBuffer);
        clSetKernelArg(activationKernel, 3, sizeof(float), &dotProduct);
        clSetKernelArg(activationKernel, 4, sizeof(int), &size);

        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &scalarSize, nullptr, 0, nullptr, nullptr);
    } else if (applyDerivativeActivation) {
        if (activationType != "linear" && activationType != "relu" &&
            activationType != "sigmoid" && activationType != "tanh") {
            Napi::TypeError::New(env, "Unsupported convolution activation derivative: " + activationType).ThrowAsJavaScriptException();
            return env.Null();
        }
        cl_kernel activationKernel = gpu.kernel("d" + activationType);
        cl_kernel elementWiseKernel = gpu.kernel("element_wise_mul");

        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &dAct);
        clSetKernelArg(activationKernel, 2, sizeof(int), &size);
        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &scalarSize, nullptr, 0, nullptr, nullptr);

        clSetKernelArg(elementWiseKernel, 0, sizeof(cl_mem), &dAct);
        clSetKernelArg(elementWiseKernel, 1, sizeof(cl_mem), &incoming_deltaBuffer);
        clSetKernelArg(elementWiseKernel, 2, sizeof(cl_mem), &finalDeltaBuffer);
        clSetKernelArg(elementWiseKernel, 3, sizeof(int), &size);

        clEnqueueNDRangeKernel(queue, elementWiseKernel, 1, nullptr, &scalarSize, nullptr, 0, nullptr, nullptr);
    } else {
        clEnqueueCopyBuffer(queue, incoming_deltaBuffer, finalDeltaBuffer, 0, 0, sizeof(float) * size, 0, nullptr, nullptr);
    }

    clSetKernelArg(gradientAccumulationKernel, 0, sizeof(cl_mem), &layerInputBuffer);
    clSetKernelArg(gradientAccumulationKernel, 1, sizeof(cl_mem), &finalDeltaBuffer);
    clSetKernelArg(gradientAccumulationKernel, 2, sizeof(cl_mem), &weightGrads);
    clSetKernelArg(gradientAccumulationKernel, 3, sizeof(cl_mem), &biasGrads);
    clSetKernelArg(gradientAccumulationKernel, 4, sizeof(int), &inputH);
    clSetKernelArg(gradientAccumulationKernel, 5, sizeof(int), &inputW);
    clSetKernelArg(gradientAccumulationKernel, 6, sizeof(int), &Cin);
    clSetKernelArg(gradientAccumulationKernel, 7, sizeof(int), &H);
    clSetKernelArg(gradientAccumulationKernel, 8, sizeof(int), &W);
    clSetKernelArg(gradientAccumulationKernel, 9, sizeof(int), &Cout);
    clSetKernelArg(gradientAccumulationKernel, 10, sizeof(int), &numFilters);
    clSetKernelArg(gradientAccumulationKernel, 11, sizeof(int), &Kh);
    clSetKernelArg(gradientAccumulationKernel, 12, sizeof(int), &Kw);
    clSetKernelArg(gradientAccumulationKernel, 13, sizeof(int), &padH);
    clSetKernelArg(gradientAccumulationKernel, 14, sizeof(int), &padW);
    clSetKernelArg(gradientAccumulationKernel, 15, sizeof(int), &stride);

    clEnqueueNDRangeKernel(queue, gradientAccumulationKernel, 1, nullptr, &gradientAccumulationSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, weightGrads, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads_array.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, biasGrads, CL_TRUE, 0, sizeof(float) * biasGradsSize, biasGrads_array.Data(), 0, nullptr, nullptr);
    
    Napi::Object output = Napi::Object::New(env);
    output.Set("delta", finalDelta);
    output.Set("weightGrads", weightGrads_array);
    output.Set("biasGrads", biasGrads_array);

    return output;

}

Napi::Value fusedFinalDeltaProjector_ForConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array zs_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array layerInput = info[1].As<Napi::Float32Array>();
    Napi::Float32Array layerOutput = info[2].As<Napi::Float32Array>();
    Napi::Float32Array incomingDelta = info[3].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads_array = info[4].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads_array = info[5].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[6].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[7].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[8].As<Napi::Array>());
    int stride = info[9].As<Napi::Number>().Int32Value();
    std::string activationType = info[10].As<Napi::String>().Utf8Value();
    bool applyDerivativeActivation = info[11].As<Napi::Boolean>().Value();
    std::string modelID = info[12].As<Napi::String>().Utf8Value();
    std::string layerID = info[13].As<Napi::String>().Utf8Value();

    if (inputShape.size() != 3 || outputShape.size() != 3 || kernelShape.size() != 4) {
        Napi::TypeError::New(env, "Convolution gradient shapes must have 3, 3, and 4 dimensions").ThrowAsJavaScriptException();
        return env.Null();
    }

    int size = incomingDelta.ElementLength();

    int numFilters = kernelShape[0];
    int Kh = kernelShape[1];
    int Kw = kernelShape[2];
    int d = kernelShape[3];
    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int Cin = inputShape[2];
    int H = outputShape[0];
    int W = outputShape[1];
    int Cout = outputShape[2];

    if (inputH <= 0 || inputW <= 0 || Cin <= 0 || H <= 0 || W <= 0 ||
        numFilters <= 0 || Kh <= 0 || Kw <= 0 || stride <= 0 ||
        d != Cin || Cout != numFilters || size != H * W * Cout ||
        zs_array.ElementLength() != size || layerOutput.ElementLength() != size ||
        layerInput.ElementLength() != inputH * inputW * Cin ||
        weightGrads_array.ElementLength() != numFilters * Kh * Kw * Cin ||
        biasGrads_array.ElementLength() != numFilters) {
        Napi::RangeError::New(env, "Convolution gradient inputs do not match the provided shapes").ThrowAsJavaScriptException();
        return env.Null();
    }

    int padH = H == inputH ? (Kh - 1) / 2 : 0;
    int padW = W == inputW ? (Kw - 1) / 2 : 0;

    Napi::Float32Array finalDelta = getOrCreateBuffer(env, modelID, layerID + "_final_delta", size);

    if (applyDerivativeActivation && activationType == "softmax") {
        DSOFTMAX(layerOutput, incomingDelta, finalDelta);
    } else if (applyDerivativeActivation) {
        using ActivationDerivative = void (*)(Napi::Float32Array&, Napi::Float32Array&);
        static const std::unordered_map<std::string, ActivationDerivative> activations = {
            {"linear", DLINEAR},
            {"relu", DRELU},
            {"sigmoid", DSIGMOID},
            {"tanh", DTANH}
        };

        auto activation = activations.find(activationType);
        if (activation == activations.end()) {
            Napi::TypeError::New(env, "Unsupported convolution activation derivative: " + activationType).ThrowAsJavaScriptException();
            return env.Null();
        }

        activation->second(zs_array, finalDelta);

        const float* incomingDeltaData = incomingDelta.Data();
        float* finalDeltaData = finalDelta.Data();
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            finalDeltaData[i] *= incomingDeltaData[i];
        }
    } else {
        const float* incomingDeltaData = incomingDelta.Data();
        float* finalDeltaData = finalDelta.Data();
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            finalDeltaData[i] = incomingDeltaData[i];
        }
    }

    float* delta = finalDelta.Data();
    const float* input = layerInput.Data();
    float* biasGrads = biasGrads_array.Data();
    float* weightGrads = weightGrads_array.Data();


    // gradient accumulation
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
    output.Set("delta", finalDelta);
    output.Set("weightGrads", weightGrads_array);
    output.Set("biasGrads", biasGrads_array);

    return output;
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

Napi::Value fusedFinalDeltaProjector_ForConv_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") return fusedFinalDeltaProjector_ForConv_GPU(info);
    return fusedFinalDeltaProjector_ForConv_CPU(info);
}

/* ==================== Module exports ======================== */
void ConvolveRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("ConvolveForward", Napi::Function::New(env, ConvolveWrapper));
    exports.Set("ConvolveBackward", Napi::Function::New(env, ConvolveDeltaWrapper));
    exports.Set("fusedFinalDeltaProjector_ForConv", Napi::Function::New(env, fusedFinalDeltaProjector_ForConv_wrapper));
}
