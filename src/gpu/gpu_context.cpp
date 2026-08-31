// src/gpu/gpu_context.cpp
#include "gpu_context.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_set>
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
    {"matmul.cl", "dot_product"},
    {"delta_matmul.cl", "delta_matmul"},
    {"convolve.cl", "convolve"},
    {"delta_convolve.cl", "delta_convolve"},
    {"maxpool.cl","maxpool"},
    {"computeBiasGradsForConnected_Layer.cl", "computeBiasGradsForConnected_Layer"},
    {"computeWeightGradsForConnected_Layer.cl", "computeWeightGradsForConnected_Layer"},
    {"computeKernelGradients.cl", "computeKernelGradients"},
    {"computeBiasGradsForConv.cl", "computeBiasGradsForConv"},
    {"activations.cl", "sigmoid"},
    {"activations.cl", "relu"},
    {"activations.cl", "Tanh"},
    {"activations.cl", "softmax"},
    {"activations.cl", "drelu"},
    {"activations.cl", "dsigmoid"},
    {"activations.cl", "dtanh"},
    {"optimizers.cl", "sgd"},
    {"optimizers.cl", "adam"},
    {"math.cl", "element_wise_mul"},
    {"math.cl", "element_wise_sub"},
    {"math.cl", "scale"},
    {"math.cl", "scale_diff"},
    {"math.cl", "accumulate_element_wise_mul"},
    {"utils.cl", "apply_padding"},
    {"utils.cl", "dilate"},
    {"maxpool.cl","maxpooldelta"},
    {"embedding.cl", "getEmbeddings"},
    {"embedding.cl", "returnEmbeddings"},
    {"loss.cl", "mse"},
    {"loss.cl", "mae"},
    {"loss.cl", "cce"},
    {"loss.cl", "bce"},
    {"normalizers.cl", "gradientClipping"},
    {"normalizers.cl", "layer_norm_standard_size"},
    {"transConv.cl", "transConv"},
    {"transConv.cl", "transConvBackward"},
    {"computeKernelGradients.cl", "accumulateTransConvKernelGrads"},
    {"attention.cl", "projectQKV"}
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

    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) { 
        errorOut = "clCreateCommandQueue failed"; 
        shutdown(); 
        return false; 
    }

    // load kernel source from kernel registry
    std::string source;

    try {
        // AFTER — track which files have already been loaded
        std::unordered_set<std::string> loadedFiles;
        for (auto& def : kernel_Definitions) {
            if (loadedFiles.count(def.file)) continue;
            loadedFiles.insert(def.file);
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

/**
 * upload output template tensors
 */
bool GpuContext::uploadOutputTemplates(const Matrix& outputs, std::string& errorOut) {
    cl_int err;

    for (auto& params : d_outputs_) clReleaseMemObject(params);

    d_outputs_.clear();

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