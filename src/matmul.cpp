// src/matmul.cpp
#include <napi.h>
#include <omp.h>
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
    int pointer = info[5].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_mem dIn  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input.Data(), nullptr);
    cl_mem dW   = gpu.getWeights(pointer);
    cl_mem dB   = gpu.getBiases(pointer);
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


    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        output[i] = b[i];
    }
    

    for (int i = 0; i < inputSize; i++) {
        float v = in[i];
        int offset = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            output[j] += v * w[offset + j];
        }
    }
    
    return outputVector;
}

Napi::Value DeltaMatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto delta   = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    int outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weightsArray = info[3].As<Napi::Float32Array>();
    int pointer = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel k = gpu.kernel("delta_matmul");

    cl_int err;
    cl_mem dDelta = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*outputSize, delta.Data(), &err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float)*inputSize, nullptr, &err);
    cl_mem dW = gpu.getWeights(pointer);

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

        #pragma omp unroll partial(4)
        for (size_t j = 0; j < outputSize; j++) {
            sum += w[offset + j] * d[j];
        }
        o[i] = sum;
    }
    return output;
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

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < inputSize; i++) {
        const float inputVal = arr1[i];
        const int rowStart = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            output[j] += inputVal * arr2[rowStart + j];
        }
    }

    return outputTensor;

}

Napi::Value ProjectOutput_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>(); // mhaOutput
    int embedDim = info[1].As<Napi::Number>().Int32Value();
    int seqLen = info[2].As<Napi::Number>().Int32Value();
    int pointer = info[3].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem dIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.ElementLength(), input.Data(), nullptr);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * input.ElementLength(), nullptr, nullptr);
    cl_mem dW = gpu.getWeights(pointer);
    cl_mem dB = gpu.getBiases(pointer);

    cl_kernel k = gpu.kernel("OutputWeightProjection");
    clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 4, sizeof(int), &embedDim);
    clSetKernelArg(k, 5, sizeof(int), &seqLen);

    size_t global = seqLen;
    clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, input.ElementLength());
    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, sizeof(float) * input.ElementLength(), output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(dIn);
    clReleaseMemObject(dOut);

    return output;
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

Napi::Value project_O_matmul(const Napi::CallbackInfo& info) {

    return ProjectOutput_GPU(info);
}

Napi::Value DotProductWrapper(const Napi::CallbackInfo& info) {
    return DotProduct_CPU(info);
}

// =================== MODULE EXPORT ===================
void MatMulRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("MatMul", Napi::Function::New(env, MatMulWrapper));
    exports.Set("DeltaMatMul", Napi::Function::New(env, DeltaMatMulWrapper));
    exports.Set("dotProduct", Napi::Function::New(env, DotProductWrapper));
    exports.Set("ProjectOutput_GPU", Napi::Function::New(env, ProjectOutput_GPU));
}
