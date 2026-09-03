#include <napi.h>
#include <omp.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <CL/cl.h>
#include <cmath>


Napi::Value GradientClipping_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputGrads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    int size = inputGrads.ElementLength();

    float* grads = inputGrads.Data();

    float norm = 0.0f;

    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += grads[i] * grads[i];
    }

    norm = std::sqrt(norm);

    if (norm > threshold) {
        float scalingVal = threshold / norm;

        auto& gpu = GpuContext::instance();
        cl_command_queue queue = gpu.queue();
        cl_context context = gpu.context();
        cl_kernel kernel = gpu.kernel("gradientClipping");

        cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inputGrads.Data(), nullptr);
        
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        clSetKernelArg(kernel, 1, sizeof(float), &scalingVal);
        clSetKernelArg(kernel, 2, sizeof(int), &size);

        size_t globalSize = (size_t)size;
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, input, CL_TRUE, 0, sizeof(int)* size, inputGrads.Data(), 0, nullptr, nullptr);
        clReleaseMemObject(input);

        return inputGrads;
    }

    return inputGrads;

}

Napi::Value GradientClipping_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputGrads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    int size = inputGrads.ElementLength();

    float* grads = inputGrads.Data();

    float norm = 0.0f;

    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += grads[i] * grads[i];
    }

    norm = std::sqrt(norm);

    if (norm > threshold) {
        float scalingVal = threshold / norm;

        #pragma omp unroll partial(4)
        for (int i = 0; i < size; i++) {
            grads[i] *= scalingVal;
        }
    }

    return inputGrads;
}

Napi::Value LayerNorm_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int size = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array gammaTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array betaTensor = info[3].As<Napi::Float32Array>();
    float eps = info[4].As<Napi::Number>().FloatValue();
    int pointer = info[5].As<Napi::Number>().Int32Value();
    std::string modelID = info[0].As<Napi::String>().Utf8Value();

    float* input = inputTensor.Data();
    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("layer_norm_standard_size");

    // compute mean
    float mean = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }

    // scale mean
    mean /= size;

    // compute variance
    float variance = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }

    // scale variance
    variance /= size;
    
    // get standardization value by getting the square root of sum of variance and epsilon
    float std = std::sqrt(variance + eps);

    cl_mem _input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* size, inputTensor.Data(), nullptr);
    cl_mem _gamma = gpu.getWeights(modelID, pointer);
    cl_mem _beta = gpu.getBiases(modelID, pointer);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* size, outputTensor.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_gamma);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_beta);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 4, sizeof(float), &mean);
    clSetKernelArg(kernel, 5, sizeof(float), &std);
    clSetKernelArg(kernel, 6, sizeof(int), &size);

    size_t globalSize = (size_t)size;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* size, outputTensor.Data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(_input);
    clReleaseMemObject(output);
    

    return outputTensor;
}

Napi::Value LayerNorm_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int size = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array gammaTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array betaTensor = info[3].As<Napi::Float32Array>();
    float eps = info[4].As<Napi::Number>().FloatValue();

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, size);

    float* input = inputTensor.Data();
    float* gamma = gammaTensor.Data();
    float* beta = betaTensor.Data();
    float* output = outputTensor.Data();

    float mean = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }

    mean /= size;

    float variance = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }

    variance /= size;

    float std = std::sqrt(variance + eps);

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float xHat = (input[i] - mean) / std;
        output[i] = gamma[i] * xHat + beta[i];
    }

    return outputTensor;
}

// ============== wrappers ==============
Napi::Value gradientClippingWrapper(const Napi::CallbackInfo& info) {
    // if (get_Global_Boolean_On_GPU()) {
    //     return GradientClipping_GPU(info);
    // }
    return GradientClipping_CPU(info);
}

Napi::Value LayerNormWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return LayerNorm_GPU(info);
    }

    return LayerNorm_CPU(info);
}


void normalizers(Napi::Env env, Napi::Object exports) {
    exports.Set("gradientClipping", Napi::Function::New(env, gradientClippingWrapper));
    exports.Set("computelayerNorm", Napi::Function::New(env, LayerNormWrapper));
}