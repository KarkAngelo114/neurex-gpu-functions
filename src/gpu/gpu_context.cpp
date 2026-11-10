// src/gpu/gpu_context.cpp
#include "gpu_context.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_set>
#include <limits>
using FloatArray = std::vector<float>;
using Matrix = std::vector<FloatArray>;

struct kernelDef {
    std::string file;
    std::string funcName;
};

struct EnumeratedDevice {
    cl_platform_id platform;
    cl_device_id   device;
};

// Future self and to others: register kernel source here.
// {`"kernel file"`,`"<kernel function name>"`}
static std::vector<kernelDef> kernel_Definitions = {
    {"matmul.cl", "matmul"},
    {"delta_matmul.cl", "delta_matmul"},
    {"convolve.cl", "convolve"},
    {"delta_convolve.cl", "delta_convolve"},
    {"maxpool.cl","maxpool"},
    {"accumulateWeightsAndBiasGradsDense.cl", "accumulateWeightsAndBiasGradsDense"},
    {"accumulateConvWeightandBiasGrads.cl", "accumulateConvWeightandBiasGrads"},
    {"activations.cl", "sigmoid_op"},
    {"activations.cl", "linear_op"},
    {"activations.cl", "relu_op"},
    {"activations.cl", "tanh_op"},
    {"activations.cl", "softmax_op"},
    {"activations.cl", "drelu"},
    {"activations.cl", "dsigmoid"},
    {"activations.cl", "dtanh"},
    {"activations.cl", "dlinear"},
    {"activations.cl", "dsoftmax"},
    {"optimizers.cl", "sgd"},
    {"optimizers.cl", "adam"},
    {"optimizers.cl", "rmsprop"},
    {"math.cl", "element_wise_mul"},
    {"math.cl", "element_wise_sub"},
    {"math.cl", "scale"},
    {"math.cl", "scale_diff"},
    {"math.cl", "accumulate_element_wise_mul"},
    {"math.cl", "element_wise_add"},
    {"maxpool.cl","maxpooldelta"},
    {"embedding.cl", "getEmbeddings"},
    {"embedding.cl", "returnEmbeddings"},
    {"loss.cl", "mse"},
    {"loss.cl", "mae"},
    {"loss.cl", "cce"},
    {"loss.cl", "bce"},
    {"normalizers.cl", "gradientClipping"},
    {"normalizers.cl", "layer_norm_standard_size"},
    {"normalizers.cl", "layer_norm_backward_one"},
    {"transConv.cl", "transConv"},
    {"transConv.cl", "transConvBackward"},
    {"accumulateConvWeightandBiasGrads.cl", "accumulateTransConvWeightAndBiasGrads"},
    {"encoding.cl", "spe"},
    {"attention.cl", "multi_head_attention"},
    {"attention.cl", "multi_head_attention_projection"},
    {"attention.cl", "multi_head_attention_backward"},
    {"attention.cl", "multi_head_attention_backward_dmha"},
    {"attention.cl", "multi_head_attention_backward_dx"},
    {"attention.cl", "accumulate_attention_weight_and_bias_grads"},
    {"gamma_and_beta_grads.cl", "accumulate_gamma_beta_grads"},
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

static std::vector<EnumeratedDevice> enumerateGPUDevices(cl_uint* platformCountOut, cl_int* errOut) {
    std::vector<EnumeratedDevice> result;
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCountOut) *platformCountOut = platformCount;
    if (errOut) *errOut = err;
    if (err != CL_SUCCESS || platformCount == 0) return result;

    std::vector<cl_platform_id> platforms(platformCount);
    err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    if (errOut) *errOut = err;
    if (err != CL_SUCCESS) return result;

    for (auto platform : platforms) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) != CL_SUCCESS
            || deviceCount == 0) continue;

        std::vector<cl_device_id> devices(deviceCount);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr) != CL_SUCCESS)
            continue;

        for (auto d : devices) result.push_back({platform, d});
    }
    return result;
}


bool GpuContext::initialize(const std::string& kernelBasePath, uint32_t deviceIndex, std::string& errorOut) {
    if (has_gpu_) return true;

    cl_int err = CL_SUCCESS;
    auto devices = enumerateGPUDevices(nullptr, &err);
    
    if (err != CL_SUCCESS || devices.empty()) {
        errorOut = "no OpenCL GPU devices available";
        return false;
    }
    if (deviceIndex >= devices.size()) {
        errorOut = "device index " + std::to_string(deviceIndex) + " out of range (found " +
                   std::to_string(devices.size()) + " devices)";
        return false;
    }

    platform_ = devices[deviceIndex].platform;
    device_   = devices[deviceIndex].device;

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

// shutdown GPU. Clears all stored clBuffers and kernels
bool GpuContext::shutdown() {
    try {
        clearAllParams();

        for (auto& kv : kernels_) clReleaseKernel(kv.second);
        kernels_.clear();
        if (program_) { clReleaseProgram(program_); program_ = nullptr; }
        if (queue_)   { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
        device_ = nullptr; platform_ = nullptr;
        has_gpu_ = false;

        return true; 
    }
    catch (...) {
        return false;
    }
    
}

// Creates a layer buffer on first use.
// On later calls, if the layer's size is unchanged, it rewrites the existing buffer in
// place with clEnqueueWriteBuffer. It only reallocates when the element count changed.
static bool upsertParamBuffer(cl_context ctx, cl_command_queue queue, std::unordered_map<std::string, cl_mem>& slots, const std::string& layerID, const FloatArray& host, std::string& errorOut) {
    const size_t bytes = sizeof(float) * host.size();
    cl_mem& slot = slots[layerID];
    if (slot) {
        size_t currentBytes = 0;
        cl_int err = clGetMemObjectInfo(slot, CL_MEM_SIZE, sizeof(currentBytes), &currentBytes, nullptr);

        if (err == CL_SUCCESS && currentBytes == bytes) {
            // Blocking write: `host` is a caller-owned reference that may be freed once we return.
            err = clEnqueueWriteBuffer(queue, slot, CL_TRUE, 0, bytes, host.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                errorOut = "Failed to write GPU buffer for layerID='" + layerID + "'.";
                return false;
            }
            return true;
        }
        // Size changed (or unreadable): drop this one buffer and fall through to recreate it.
        clReleaseMemObject(slot);
        slot = nullptr;
    }

    cl_int err = CL_SUCCESS;
    slot = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, const_cast<float*>(host.data()), &err);
    
    if (err != CL_SUCCESS || slot == nullptr) {
        slot = nullptr;
        errorOut = "Failed to allocate GPU buffer for layerID='" + layerID + "'.";
        return false;
    }
    return true;
}

bool GpuContext::uploadParams(const std::string& modelID, const std::vector<LayerParameters>& parameters, std::string& errorOut) {
    if (!has_gpu_) {
        errorOut = "GPU context is not initialized.";
        return false;
    }


    auto& weights = weightsByModel_[modelID];
    auto& biases = biasesByModel_[modelID];
    std::unordered_set<std::string> layerIDs;

    for (const auto& layer : parameters) {
        if (layer.layerID.empty() || !layerIDs.insert(layer.layerID).second) {
            errorOut = "Parameter metadata contains a missing or duplicate layerID.";
            return false;
        }
        if (!upsertParamBuffer(context_, queue_, weights, layer.layerID, layer.weights, errorOut) ||
            !upsertParamBuffer(context_, queue_, biases, layer.layerID, layer.biases, errorOut)) {
            return false;
        }
    }

    auto removeStaleBuffers = [&layerIDs](auto& buffers) {
        for (auto it = buffers.begin(); it != buffers.end();) {
            if (!layerIDs.count(it->first)) {
                if (it->second) clReleaseMemObject(it->second);
                it = buffers.erase(it);
            } else {
                ++it;
            }
        }
    };
    removeStaleBuffers(weights);
    removeStaleBuffers(biases);

    return true;
}
void GpuContext::clearParams(const std::string& modelID) {
    auto itWeights = weightsByModel_.find(modelID);
    if (itWeights != weightsByModel_.end()) {
        for (const auto& entry : itWeights->second) if (entry.second) clReleaseMemObject(entry.second);
        weightsByModel_.erase(itWeights);
    }

    auto itBiases = biasesByModel_.find(modelID);
    if (itBiases != biasesByModel_.end()) {
        for (const auto& entry : itBiases->second) if (entry.second) clReleaseMemObject(entry.second);
        biasesByModel_.erase(itBiases);
    }

    // NOTE: optimizer states (m/v/velocity/sqAvg) are intentionally NOT cleared here.
    // clearParams() is called from two places with two different intents:
    //   - uploadParams(): re-syncing the SAME model's weights/biases mid-training.
    //     The optimizer states must survive this, otherwise every training step
    //     would reset them to zero and the whole point of caching them is lost.
    //   - ReleaseParams() (JS-facing): the model is actually being torn down.
    //     That call site explicitly calls clearOptimizerStates() itself; see globals.cpp.
    //
    // The forward/backward activation caches (z/activation_output/dAct/delta) are
    // ALSO intentionally not cleared here, for a different reason: they're overwritten
    // every single feedforward/backprop pass by design (see getOrCreateCacheBuffer),
    // so leaving stale data behind after a mid-training uploadParams() is harmless —
    // it'll be overwritten before anything ever reads it again. Freeing and
    // reallocating them here would just be wasted churn since layer shapes (and thus
    // buffer sizes) don't change between re-syncs. Like optimizer states, they're only
    // released on full teardown; see clearActivationCaches().
}

/** 
 * this function is full tear-down of the GPU lifecycle.
 * if a `neurex` instance calls `shutdown()`, other instances will also be affected.
 */
void GpuContext::clearAllParams() {
    auto releaseBuffers = [](const BufferCache& cache) {
        for (const auto& modelEntry : cache) {
            for (const auto& bufferEntry : modelEntry.second) {
                if (bufferEntry.second) clReleaseMemObject(bufferEntry.second);
            }
        }
    };
    releaseBuffers(weightsByModel_);
    releaseBuffers(biasesByModel_);
    auto releaseCache = [](const BufferCache& cache) {
        for (const auto& modelEntry : cache) {
            for (const auto& bufferEntry : modelEntry.second) {
                if (bufferEntry.second) clReleaseMemObject(bufferEntry.second);
            }
        }
    };
    releaseCache(zByModel_);
    releaseCache(activationOutputsByModel_);
    releaseCache(dActByModel_);
    releaseCache(deltasByModel_);
    releaseCache(dBetaByModel_);
    releaseCache(dGammaByModel_);
    releaseCache(inputsByModel_); 
    releaseCache(somethingToWriteOn_);
    releaseBuffers(mStatesWeights_);
    releaseBuffers(mStatesBiases_);
    releaseBuffers(vStatesWeights_);
    releaseBuffers(vStatesBiases_);
    releaseBuffers(velocityWeights_);
    releaseBuffers(velocityBiases_);
    releaseBuffers(sqAvgWeights_);
    releaseBuffers(sqAvgBiases_);
    
    weightsByModel_.clear();
    biasesByModel_.clear();
    zByModel_.clear();
    inputsByModel_.clear();
    activationOutputsByModel_.clear();
    dActByModel_.clear();
    deltasByModel_.clear();
    dBetaByModel_.clear();
    dGammaByModel_.clear();
    somethingToWriteOn_.clear();
    mStatesWeights_.clear(); 
    mStatesBiases_.clear();
    vStatesWeights_.clear(); 
    vStatesBiases_.clear();
    velocityWeights_.clear(); 
    velocityBiases_.clear();
    sqAvgWeights_.clear(); 
    sqAvgBiases_.clear();
}

cl_kernel GpuContext::kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return it == kernels_.end() ? nullptr : it->second;
}

// ===================== Optimizer state caching =====================

cl_mem GpuContext::getOrCreateStateBuffer(BufferCache& store, const std::string& modelID, const std::string& layerID, size_t length, const float* initialData) {
    if (layerID.empty()) {
        throw std::invalid_argument("GpuContext: optimizer state requires a layerID.");
    }

    auto& layerBuffers = store[modelID];
    auto found = layerBuffers.find(layerID);
    if (found != layerBuffers.end()) {
        size_t currentSize = 0;
        cl_int err = clGetMemObjectInfo(found->second, CL_MEM_SIZE, sizeof(currentSize), &currentSize, nullptr);
        if (err == CL_SUCCESS && currentSize >= sizeof(float) * length) {
            return found->second;
        }
        if (found->second) clReleaseMemObject(found->second);
        layerBuffers.erase(found);
    }

    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * length, const_cast<float*>(initialData), &err);
    if (err != CL_SUCCESS || buffer == nullptr) {
        throw std::runtime_error("GpuContext: failed to allocate optimizer state for layerID='" + layerID + "'.");
    }
    layerBuffers.emplace(layerID, buffer);
    return buffer;
}

cl_mem GpuContext::getOrCreateCacheBuffer(BufferCache& store, const std::string& modelID, const std::string& layerID, size_t length) {
    if (layerID.empty()) {
        throw std::invalid_argument("GpuContext: layerID must not be empty.");
    }
    if (length == 0 || length > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("GpuContext: cached buffer length must be positive and fit in bytes.");
    }

    auto& layerBuffers = store[modelID];
    auto bufferIt = layerBuffers.find(layerID);
    if (bufferIt != layerBuffers.end()) {
        size_t currentSize = 0;
        cl_int err = clGetMemObjectInfo(bufferIt->second, CL_MEM_SIZE, sizeof(currentSize), &currentSize, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("GpuContext: failed to inspect cached buffer for modelID='" + modelID + "' layerID='" + layerID + "'.");
        }
        if (currentSize >= sizeof(float) * length) {
            return bufferIt->second;
        }
    }

    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(float) * length, nullptr, &err);
    if (err != CL_SUCCESS || buffer == nullptr) {
        if (buffer) clReleaseMemObject(buffer);
        throw std::runtime_error("GpuContext: failed to allocate cached buffer for modelID='" + modelID + "' layerID='" + layerID + "'.");
    }

    if (bufferIt == layerBuffers.end()) {
        layerBuffers.emplace(layerID, buffer);
    } else {
        cl_mem oldBuffer = bufferIt->second;
        bufferIt->second = buffer;
        if (oldBuffer) clReleaseMemObject(oldBuffer);
    }
    return buffer;
}

cl_mem GpuContext::getOrCreate_M(const std::string& modelID, const std::string& layerID, bool isWeights, size_t length, const float* initialData) {
    return getOrCreateStateBuffer(isWeights ? mStatesWeights_ : mStatesBiases_, modelID, layerID, length, initialData);
}

cl_mem GpuContext::getOrCreate_V(const std::string& modelID, const std::string& layerID, bool isWeights, size_t length, const float* initialData) {
    return getOrCreateStateBuffer(isWeights ? vStatesWeights_ : vStatesBiases_, modelID, layerID, length, initialData);
}

cl_mem GpuContext::getOrCreate_Velocity(const std::string& modelID, const std::string& layerID, bool isWeights, size_t length, const float* initialData) {
    return getOrCreateStateBuffer(isWeights ? velocityWeights_ : velocityBiases_, modelID, layerID, length, initialData);
}

cl_mem GpuContext::getOrCreate_SqAvg(const std::string& modelID, const std::string& layerID, bool isWeights, size_t length, const float* initialData) {
    return getOrCreateStateBuffer(isWeights ? sqAvgWeights_ : sqAvgBiases_, modelID, layerID, length, initialData);
}

static void releaseAndClearModelEntry(BufferCache& store, const std::string& modelID) {
    auto it = store.find(modelID);
    if (it != store.end()) {
        for (const auto& entry : it->second) {
            if (entry.second) clReleaseMemObject(entry.second);
        }
        store.erase(it);
    }
}

// ===================== Forward/backward activation caching =====================

cl_mem GpuContext::getOrCreate_Z(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(zByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_ActivationOutput(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(activationOutputsByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_DAct(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(dActByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_Delta(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(deltasByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_dBeta(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(dBetaByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_dGamma(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(dGammaByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_Input(const std::string& modelID, const std::string& layerID, size_t length) {
    return getOrCreateCacheBuffer(inputsByModel_, modelID, layerID, length);
}

cl_mem GpuContext::getOrCreate_SomethingToWriteOn(const std::string& globalReferenceID, const std::string& localReferenceID, size_t length) {
    return getOrCreateCacheBuffer(somethingToWriteOn_, globalReferenceID, localReferenceID, length);
}

void GpuContext::clearOptimizerStates(const std::string& modelID) {
    releaseAndClearModelEntry(mStatesWeights_, modelID);
    releaseAndClearModelEntry(mStatesBiases_, modelID);
    releaseAndClearModelEntry(vStatesWeights_, modelID);
    releaseAndClearModelEntry(vStatesBiases_, modelID);
    releaseAndClearModelEntry(velocityWeights_, modelID);
    releaseAndClearModelEntry(velocityBiases_, modelID);
    releaseAndClearModelEntry(sqAvgWeights_, modelID);
    releaseAndClearModelEntry(sqAvgBiases_, modelID);
}

// Releases this model's cached z/activation_output/dAct/delta buffers. Like
// clearOptimizerStates(), this is for full model teardown — call this alongside
// clearOptimizerStates() from the same JS-facing ReleaseParams() call site, NOT from
// clearParams()/uploadParams(); see the note in clearParams() for why.
void GpuContext::clearActivationCaches(const std::string& modelID) {
    releaseAndClearModelEntry(zByModel_, modelID);
    releaseAndClearModelEntry(activationOutputsByModel_, modelID);
    releaseAndClearModelEntry(dActByModel_, modelID);
    releaseAndClearModelEntry(deltasByModel_, modelID);
    releaseAndClearModelEntry(inputsByModel_, modelID);
    releaseAndClearModelEntry(somethingToWriteOn_, modelID);
}

void GpuContext::clear_dBeta_And_dGamma_By_Model(const std::string& modelID) {
    releaseAndClearModelEntry(dBetaByModel_, modelID);
    releaseAndClearModelEntry(dGammaByModel_, modelID);
}