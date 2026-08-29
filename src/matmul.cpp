// src/matmul.cpp
#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <algorithm>
#include <vector>
#include "functions/functions.h"

using Array = std::vector<float>;

// ============== MatMul ==============

Napi::Value MatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int inputSize = info[1].As<Napi::Number>().Int32Value();
    int outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[4].As<Napi::Float32Array>();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_mem dIn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input.Data(), nullptr);
    cl_mem dW   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weights.ElementLength(), weights.Data(), nullptr);
    cl_mem dB   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * biases.ElementLength(), biases.Data(), nullptr);
    cl_mem dOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputSize, nullptr, nullptr);

    cl_kernel k = gpu.kernel("matmul");

    clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 4, sizeof(int), &inputSize);
    clSetKernelArg(k, 5, sizeof(int), &outputSize);

    size_t global = outputSize;
    clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);
    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, sizeof(float) * outputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(dIn);
    clReleaseMemObject(dW);
    clReleaseMemObject(dB);
    clReleaseMemObject(dOut);

    return output;
}

Napi::Value MatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    int outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[4].As<Napi::Float32Array>();

    Napi::Float32Array outputVector = Napi::Float32Array::New(env, outputSize);

    float* in = input.Data();
    float* w = weights.Data();
    float* b = biases.Data();
    float* output = outputVector.Data();

    for (int i = 0; i < outputSize; i++) {
        output[i] = b[i];
    }
    

    for (int i = 0; i < inputSize; i++) {
        float v = in[i];
        int offset = i * outputSize;
        for (int j = 0; j < outputSize; j++) {
            output[j] += v * w[offset + j];
        }
    }
    
    return outputVector;
}

// ============== DeltaMatMul ==============

Napi::Value DeltaMatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto delta   = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    int outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weightsArray = info[3].As<Napi::Float32Array>();

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel k = gpu.kernel("delta_matmul");

    cl_int err;
    cl_mem dDelta = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*outputSize, delta.Data(), &err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*inputSize, nullptr, &err);
    cl_mem dW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightsArray.ElementLength(), weightsArray.Data(), nullptr);

    clSetKernelArg(k, 0, sizeof(cl_mem), &dDelta);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 3, sizeof(int),&inputSize);
    clSetKernelArg(k, 4, sizeof(int), &outputSize);

    size_t global = (size_t)inputSize;
    clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, inputSize);
    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, sizeof(float)*inputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(dDelta);
    clReleaseMemObject(dOut);
    clReleaseMemObject(dW);

    return output;
}

Napi::Value DeltaMatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array delta = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    size_t outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    Napi::Float32Array output = Napi::Float32Array::New(env, inputSize);

    float* d = delta.Data();
    const float* w = weights.Data();
    float* o = output.Data();

    for (int i = 0; i < inputSize; i++) {
        float sum = 0.0f;
        size_t offset = i * outputSize;
        for (size_t j = 0; j < outputSize; j++) {
            sum += w[offset + j] * d[j];
        }
        o[i] = sum;
    }
    return output;
}

Napi::Value DotProduct_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1_input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2_input = info[1].As<Napi::Float32Array>();
    int inputSize = info[2].As<Napi::Number>().Int32Value();
    int outputSize = info[3].As<Napi::Number>().Int32Value();

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dot_product");

    cl_mem inputArr1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* arr1_input.ElementLength(), arr1_input.Data(), nullptr);
    cl_mem inputArr2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* arr2_input.ElementLength(), arr2_input.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* outputSize, outputTensor.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputArr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputArr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &inputSize);
    clSetKernelArg(kernel, 4, sizeof(int), &outputSize);

    size_t globalSize = (size_t)outputSize;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* outputSize, outputTensor.Data(), 0, nullptr, nullptr );

    clReleaseMemObject(inputArr1);
    clReleaseMemObject(inputArr2);
    clReleaseMemObject(output);

    return outputTensor;
}

Napi::Value DotProduct_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1_input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2_input = info[1].As<Napi::Float32Array>();
    int inputSize = info[2].As<Napi::Number>().Int32Value();
    int outputSize = info[3].As<Napi::Number>().Int32Value();

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);

    float* arr1 = arr1_input.Data();
    float* arr2 = arr2_input.Data();
    float* output = outputTensor.Data();

    for (int i = 0; i < inputSize; i++) {
        const float inputVal = arr1[i];
        const int rowStart = i * outputSize;

        for (int j = 0; j < outputSize; j++) {
            output[j] += inputVal * arr2[rowStart + j];
        }
    }

    return outputTensor;

}

// ================== Wrappers ====================== //
Napi::Value MatMulWrapper(const Napi::CallbackInfo& info) {

    if (get_Global_Boolean_On_GPU()) {
        return MatMul_GPU(info);
    }

    return MatMul_CPU(info);
}

Napi::Value DeltaMatMulWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DeltaMatMul_GPU(info);
    }
    return DeltaMatMul_CPU(info);
}

Napi::Value DotProductWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DotProduct_GPU(info);
    }
    return DotProduct_CPU(info);
}

// =================== MODULE EXPORT ===================
void MatMulRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("MatMul", Napi::Function::New(env, MatMulWrapper));
    exports.Set("DeltaMatMul", Napi::Function::New(env, DeltaMatMulWrapper));
    exports.Set("dotProduct", Napi::Function::New(env, DotProductWrapper));
}
