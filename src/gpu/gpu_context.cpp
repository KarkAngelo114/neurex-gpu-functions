// src/gpu/gpu_context.cpp
#include "gpu_context.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
using FloatArray = std::vector<float>;
using Matrix = std::vector<FloatArray>;

struct kernelDef {
    std::string file;
    std::string funcName;
};

// Future self and to others: register kernel source here.
// {`"kernel file"`,`"<kernel function name>"`}
static std::vector<kernelDef> kernel_Definitions = {
    {"matmul.cl", "matmul"},
    {"delta_matmul.cl", "delta_matmul"},
    {"convolve.cl", "convolve"},
    {"delta_convolve.cl", "delta_convolve"},
    {"scaleGrads.cl", "scaleGrads"},
    {"maxpool.cl","maxpool"}
};


GpuContext& GpuContext::instance() {
    static GpuContext g;
    return g;
}

static std::string pathResolver(const std::string& relative) {
    return std::filesystem::absolute(relative).string();
}


// load kernel source from file, accepts string file path
static std::string LoadKernelFile(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("An error occured opening kernel source:"+path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool GpuContext::initialize(const std::string& kernelBasePath, std::string& errorOut) {
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
    if (!device_) {
        errorOut = "no GPU device"; 
        return false; 
    }

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        errorOut = "clCreateContext failed"; 
        return false;
    }

    // OpenCL 1.2 path; if you target 2.0+ use clCreateCommandQueueWithProperties.
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) { 
        errorOut = "clCreateCommandQueue failed"; 
        shutdown(); 
        return false; 
    }

    // load kernel source from kernel registry
    std::string source;

    try {
        for (auto& def : kernel_Definitions) {
            std::filesystem::path fullPath = std::filesystem::path(kernelBasePath) / def.file;
            source += LoadKernelFile(fullPath.string()) + "\n";
        }
    } catch (const std::exception& e) {
        errorOut = e.what();
        return false;
    }

    const char* kernel_source_code = source.c_str();

    program_ = clCreateProgramWithSource(context_, 1, &kernel_source_code, nullptr, &err);
    if (err != CL_SUCCESS) { 
        errorOut = "clCreateProgramWithSource failed"; 
        shutdown(); 
        return false; 
    }

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

    // for future me and others, register kernel names here after creating an OpenCL C function
    for (auto& kernel : kernel_Definitions) {
        cl_kernel k = clCreateKernel(program_, kernel.funcName.c_str(), &err);
        if (err != CL_SUCCESS || k == nullptr) {
            errorOut = "clCreateKernel failed: " + kernel.funcName +". ";
            shutdown();
            return false;
        }
        kernels_[kernel.funcName] = k;
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

bool GpuContext::uploadParams(
    const Matrix& weights,
    const Matrix& biases,
    const Matrix& outputs,
    std::string& errorOut
) {
    cl_int err;

    // release old buffers (if re-uploading)
    for (auto& params : d_weights_) clReleaseMemObject(params);
    for (auto& params : d_biases_)  clReleaseMemObject(params);
    for (auto& params : d_outputs_) clReleaseMemObject(params);

    d_weights_.clear();
    d_biases_.clear();
    d_outputs_.clear();

    // upload weights
    for (const auto& w : weights) {
        cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * w.size(), (void*)w.data(), &err);
        if (err != CL_SUCCESS) {
            errorOut = "upload weights failed";
            return false;
        }
        d_weights_.push_back(buf);
    }

    // upload biases
    for (const auto& b : biases) {
        cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * b.size(), (void*)b.data(), &err);
        if (err != CL_SUCCESS) {
            errorOut = "upload biases failed";
            return false;
        }
        d_biases_.push_back(buf);
    }

    // allocate outputs (no host copy needed)
    for (const auto& o : outputs) {
        cl_mem buf = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(float) * o.size(), nullptr, &err);
        if (err != CL_SUCCESS) {
            errorOut = "alloc outputs failed";
            return false;
        }
        d_outputs_.push_back(buf);
    }

    return true;
}

cl_kernel GpuContext::kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return it == kernels_.end() ? nullptr : it->second;
}