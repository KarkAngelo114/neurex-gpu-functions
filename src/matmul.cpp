// src/matmul.cpp
#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include <algorithm>

// ============== MatMul ==============

static Napi::Value MatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto input    = info[0].As<Napi::Float32Array>();
    auto weights  = info[1].As<Napi::Float32Array>();
    auto biases   = info[2].As<Napi::Float32Array>();
    int inputSize  = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue q = gpu.queue();
    cl_kernel k = gpu.kernel("matmul");

    cl_int err;
    cl_mem dIn  = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float)*inputSize, input.Data(), &err);
    cl_mem dW   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float)*inputSize*outputSize, weights.Data(), &err);
    cl_mem dB   = clCreateBuffer(ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float)*outputSize, biases.Data(), &err);
    cl_mem dOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 sizeof(float)*outputSize, nullptr, &err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &dIn);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 4, sizeof(int),    &inputSize);
    clSetKernelArg(k, 5, sizeof(int),    &outputSize);

    size_t global = (size_t)outputSize;
    clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);
    clEnqueueReadBuffer(q, dOut, CL_TRUE, 0, sizeof(float)*outputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(dIn);
    clReleaseMemObject(dW);
    clReleaseMemObject(dB);
    clReleaseMemObject(dOut);
    return output;
}

static Napi::Value MatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input   = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases  = info[2].As<Napi::Float32Array>();
    int inputSize  = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);
    float* in = input.Data();
    float* w  = weights.Data();
    float* b  = biases.Data();
    float* x  = output.Data();

    std::copy(b, b + outputSize, x);

    for (int i = 0; i < inputSize; i++) {
        float v = in[i];
        int offset = i * outputSize;
        for (int j = 0; j < outputSize; j++) {
            x[j] += v * w[offset + j];
        }
    }
    return output;
}

static Napi::Value MatMulWrapper(const Napi::CallbackInfo& info) {
    if (GpuContext::instance().hasGPU()) return MatMul_GPU(info);
    return MatMul_CPU(info);
}

// ============== DeltaMatMul ==============

static Napi::Value DeltaMatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto delta   = info[0].As<Napi::Float32Array>();
    auto weights = info[1].As<Napi::Float32Array>();
    int inputSize  = info[2].As<Napi::Number>().Int32Value();
    int outputSize = info[3].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue q = gpu.queue();
    cl_kernel k = gpu.kernel("delta_matmul");

    cl_int err;
    cl_mem dDelta = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)*outputSize, delta.Data(), &err);
    cl_mem dW     = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float)*inputSize*outputSize, weights.Data(), &err);
    cl_mem dOut   = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                   sizeof(float)*inputSize, nullptr, &err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &dDelta);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 3, sizeof(int),    &inputSize);
    clSetKernelArg(k, 4, sizeof(int),    &outputSize);

    size_t global = (size_t)inputSize;
    clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, inputSize);
    clEnqueueReadBuffer(q, dOut, CL_TRUE, 0, sizeof(float)*inputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(dDelta);
    clReleaseMemObject(dW);
    clReleaseMemObject(dOut);
    return output;
}

static Napi::Value DeltaMatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array delta   = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
    int inputSize  = info[2].As<Napi::Number>().Int32Value();
    int outputSize = info[3].As<Napi::Number>().Int32Value();

    Napi::Float32Array output = Napi::Float32Array::New(env, inputSize);
    float* d = delta.Data();
    float* w = weights.Data();
    float* o = output.Data();

    for (int i = 0; i < inputSize; i++) {
        float sum = 0.0f;
        int offset = i * outputSize;
        for (int j = 0; j < outputSize; j++) {
            sum += w[offset + j] * d[j];
        }
        o[i] = sum;
    }
    return output;
}

static Napi::Value DeltaMatMulWrapper(const Napi::CallbackInfo& info) {
    if (GpuContext::instance().hasGPU()) return DeltaMatMul_GPU(info);
    return DeltaMatMul_CPU(info);
}

// =================== MODULE EXPORT ===================
void MatMulRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("MatMul",      Napi::Function::New(env, MatMulWrapper));
    exports.Set("DeltaMatMul", Napi::Function::New(env, DeltaMatMulWrapper));
}