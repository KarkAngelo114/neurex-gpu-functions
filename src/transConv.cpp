#include <napi.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "global_activations/global_activations.h"
#include <unordered_map>


using IntArray = std::vector<int>;


static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

Napi::Value transConvForward_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    std::string activationType = info[6].As<Napi::String>().Utf8Value();
    int pointer = info[9].As<Napi::Number>().Int32Value();
    std::string modelID = info[10].As<Napi::String>().Utf8Value();
    std::string layerID = info[12].As<Napi::String>().Utf8Value()

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
    int inputSize = inputTensor.ElementLength();

    size_t outputSize = oH * oW * f;
    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_transConv_forward_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_transConv_forward_activated_output", outputSize);

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel transConvKernel = gpu.kernel("transConv");
    cl_kernel activationKernel = gpu.kernel(activationType+"_op");

    cl_mem input = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(inputSize));
    clEnqueueWriteBuffer(queue, input, CL_FALSE, 0, sizeof(float) * inputSize, inputTensor.Data(), 0, nullptr, nullptr);
    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem biases = gpu.getBiases(modelID, pointer);
    cl_mem z_values = gpu.getOrCreate_Z(modelID, layerID, static_cast<size_t>(outputSize));
    cl_mem activated_output = gpu.getOrCreate_ActivationOutput(modelID, layerID, static_cast<size_t>(outputSize));

    clSetKernelArg(transConvKernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(transConvKernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(transConvKernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(transConvKernel, 3, sizeof(cl_mem), &z_values);
    clSetKernelArg(transConvKernel, 4, sizeof(int), &iH);
    clSetKernelArg(transConvKernel, 5, sizeof(int), &iW);
    clSetKernelArg(transConvKernel, 6, sizeof(int), &iD);
    clSetKernelArg(transConvKernel, 7, sizeof(int), &oH);
    clSetKernelArg(transConvKernel, 8, sizeof(int), &oW);
    clSetKernelArg(transConvKernel, 9, sizeof(int), &f);
    clSetKernelArg(transConvKernel, 10, sizeof(int), &kh);
    clSetKernelArg(transConvKernel, 11, sizeof(int), &kw);
    clSetKernelArg(transConvKernel, 12, sizeof(int), &d);
    clSetKernelArg(transConvKernel, 13, sizeof(int), &strides);
    clSetKernelArg(transConvKernel, 14, sizeof(int), &padTop);
    clSetKernelArg(transConvKernel, 15, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        (size_t)oH,
        (size_t)oW,
        (size_t)f
    };

    clEnqueueNDRangeKernel(queue, transConvKernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
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

    Napi::Object output = Napi::Object::New(env);
    output.Set("z_values", z_values_vector);
    output.Set("activated_output",activated_output_vector);

    return output;
}

Napi::Value transConvForward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    std::string activationType = info[6].As<Napi::String>().Utf8Value();
    Napi::Float32Array weightsArray = info[7].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[8].As<Napi::Float32Array>();
    int pointer = info[9].As<Napi::Number>().Int32Value();
    std::string modelID = info[10].As<Napi::String>().Utf8Value();
    std::string layerID = info[12].As<Napi::String>().Utf8Value();

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
    int size = oH * oW * f;

    std::unordered_map<std::string, void(*)(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn)> activations = {
        {"linear", LINEAR},
        {"relu", RELU},
        {"sigmoid", SIGMOID},
        {"tanh", TANH},
        {"softmax", SOFTMAX}
    };

    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_transConv_forward_z_values", size);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_transConv_forward_activated_output", size);

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
    float* output = z_values_vector.Data();

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

    activations[activationType](z_values_vector, activated_output_vector);

    Napi::Object output = Napi::Object::New(env);
    output.Set("z_values", z_values_vector);
    output.Set("activated_output",activated_output_vector);

    return output;
}

Napi::Value transConvBackward_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();
    int pointer = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

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
    int size = iH * iW * iD;
    Napi::Float32Array outputTensor = getOrCreateBuffer(env, modelID, layerID+"transConv_backward", size);

    int deltaSize = deltaTensor.ElementLength();
    int outputSize = outputTensor.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("transConvBackward");

    cl_mem delta = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_trans_conv_incoming_delta", static_cast<size_t>(deltaSize));
    clEnqueueWriteBuffer(queue, delta, CL_FALSE, 0, sizeof(float) * deltaSize, deltaTensor.Data(), 0, nullptr, nullptr);
    cl_mem output = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_trans_conv_backward_output", static_cast<size_t>(outputSize));
    cl_mem weights = gpu.getWeights(modelID, pointer);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &iH);
    clSetKernelArg(kernel, 4, sizeof(int), &iW);
    clSetKernelArg(kernel, 5, sizeof(int), &iD);
    clSetKernelArg(kernel, 6, sizeof(int), &oH);
    clSetKernelArg(kernel, 7, sizeof(int), &oW);
    clSetKernelArg(kernel, 8, sizeof(int), &oD);
    clSetKernelArg(kernel, 9, sizeof(int), &f);
    clSetKernelArg(kernel, 10, sizeof(int), &kh);
    clSetKernelArg(kernel, 11, sizeof(int), &kw);
    clSetKernelArg(kernel, 12, sizeof(int), &d);
    clSetKernelArg(kernel, 13, sizeof(int), &strides);
    clSetKernelArg(kernel, 14, sizeof(int), &padTop);
    clSetKernelArg(kernel, 15, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        (size_t)iH,
        (size_t)iW,
        (size_t)iD
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * outputTensor.ElementLength(), outputTensor.Data(), 0, nullptr, nullptr);

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
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

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
    int size = iH * iW * iD;
    Napi::Float32Array outputTensor = getOrCreateBuffer(env, modelID ,layerID+"_transConv_backward" ,size);

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

Napi::Value transConvForwardWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return transConvForward_GPU(info);
    }
    return transConvForward_CPU(info);
}

Napi::Value transConvBackwardWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return transConvBackward_GPU(info);
    }
    return transConvBackward_CPU(info);
}

void transConvFunc(Napi::Env env, Napi::Object exports) {
    exports.Set("transConvForward", Napi::Function::New(env, transConvForwardWrapper));
    exports.Set("transConvBackward", Napi::Function::New(env, transConvBackwardWrapper));
}