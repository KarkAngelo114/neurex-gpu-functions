// src/gpu/gpu_context.cpp
#include "gpu_context.h"
#include <vector>

static const char* kKernelSrc = R"CLC(
__kernel void matmul(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    const int inputSize,
    const int outputSize)
{
    int j = get_global_id(0);
    if (j >= outputSize) return;

    float acc = biases[j];
    for (int i = 0; i < inputSize; ++i) {
        acc += input[i] * weights[i * outputSize + j];
    }
    output[j] = acc;
}

__kernel void delta_matmul(
    __global const float* delta,
    __global const float* weights,
    __global float* output,
    const int inputSize,
    const int outputSize)
{
    int i = get_global_id(0);
    if (i >= inputSize) return;

    float sum = 0.0f;
    int offset = i * outputSize;
    for (int j = 0; j < outputSize; ++j) {
        sum += weights[offset + j] * delta[j];
    }
    output[i] = sum;
}
)CLC";

GpuContext& GpuContext::instance() {
    static GpuContext g;
    return g;
}

bool GpuContext::initialize(std::string& errorOut) {
    if (has_gpu_) return true; // idempotent

    cl_int err;

    // Pick first GPU on first platform that has one.
    cl_uint platCount = 0;
    clGetPlatformIDs(0, nullptr, &platCount);
    if (platCount == 0) { 
        errorOut = "no OpenCL platforms"; 
        return false; 
    }

    std::vector<cl_platform_id> plats(platCount);
    clGetPlatformIDs(platCount, plats.data(), nullptr);

    for (auto p : plats) {
        cl_uint dCount = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &dCount) != CL_SUCCESS) continue;
        if (dCount == 0) continue;
        std::vector<cl_device_id> devs(dCount);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, dCount, devs.data(), nullptr);
        platform_ = p;
        device_   = devs[0];
        break;
    }
    if (!device_) { errorOut = "no GPU device"; return false; }

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { errorOut = "clCreateContext failed"; return false; }

    // OpenCL 1.2 path; if you target 2.0+ use clCreateCommandQueueWithProperties.
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) { errorOut = "clCreateCommandQueue failed"; shutdown(); return false; }

    program_ = clCreateProgramWithSource(context_, 1, &kKernelSrc, nullptr, &err);
    if (err != CL_SUCCESS) { errorOut = "clCreateProgramWithSource failed"; shutdown(); return false; }

    err = clBuildProgram(program_, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        errorOut = "clBuildProgram failed: " + log;
        shutdown();
        return false;
    }

    for (const char* name : {"matmul", "delta_matmul"}) {
        cl_kernel k = clCreateKernel(program_, name, &err);
        if (err != CL_SUCCESS) { errorOut = std::string("clCreateKernel failed: ") + name; shutdown(); return false; }
        kernels_[name] = k;
    }

    has_gpu_ = true;
    return true;
}

void GpuContext::shutdown() {
    for (auto& kv : kernels_) clReleaseKernel(kv.second);
    kernels_.clear();
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
    if (queue_)   { clReleaseCommandQueue(queue_); queue_ = nullptr; }
    if (context_) { clReleaseContext(context_); context_ = nullptr; }
    device_ = nullptr; platform_ = nullptr;
    has_gpu_ = false;
}

cl_kernel GpuContext::kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return it == kernels_.end() ? nullptr : it->second;
}