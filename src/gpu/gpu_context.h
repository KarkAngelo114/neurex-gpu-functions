#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
using FloatArray = std::vector<float>;
struct LayerParameters {
    std::string layerID;
    FloatArray weights;
    FloatArray biases;
};
using BufferCache = std::unordered_map<std::string, std::unordered_map<std::string, cl_mem>>;

// Looks up a keyed cache entry and reports its cache, model, and key on failure.
inline cl_mem lookupCachedBuffer(const BufferCache& store, const std::string& cacheName, const std::string& modelID, const std::string& layerID) {
    if (layerID.empty()) {
        throw std::invalid_argument("GpuContext: layerID must not be empty.");
    }
    auto modelIt = store.find(modelID);
    if (modelIt == store.end()) {
        throw std::out_of_range("GpuContext: no '" + cacheName + "' cache entries for modelID='" + modelID + "' layerID='" + layerID + "'.");
    }
    auto bufferIt = modelIt->second.find(layerID);
    if (bufferIt == modelIt->second.end()) {
        throw std::out_of_range("GpuContext: '" + cacheName + "' has no buffer for modelID='" + modelID + "' layerID='" + layerID + "'.");
    }
    return bufferIt->second;
}


class GpuContext {
    public:
        static GpuContext& instance();

        bool initialize(const std::string& kernelBasePath, uint32_t deviceIndex, std::string& errorOut);

        // shuts down the GPU and clearing all clBuffers, context, kernels, devices and platforms
        bool shutdown();

        bool hasGPU() { 
            return has_gpu_; 
        }

        // upload model parameters each referenced with a model ID
        bool uploadParams(const std::string& modelID, const std::vector<LayerParameters>& parameters, std::string& errorOut);

        // release one model's buffers
        void clearParams(const std::string& modelID);

        // release everything (used by shutdown())
        void clearAllParams();

        /**
         * fetches the corresponding weights for the current layer, scoped to modelID
         * @param modelID use to reference what model's weights will get
         * @param layerID identifies the layer that owns the parameters
         * @return a clBuffer
         */
        cl_mem getWeights(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(weightsByModel_, "weightsByModel_", modelID, layerID);
        }

        /**
         * fetches the corresponding biases for the current layer, scoped to modelID
         * @param modelID use to reference what model's weights will get
         * @param layerID identifies the layer that owns the parameters
         * @return a clBuffer
         */
        cl_mem getBiases(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(biasesByModel_, "biasesByModel_", modelID, layerID);
        }

        // release every cached activation buffer for one model (called from ReleaseParams)
        void clearActivationCaches(const std::string& modelID);

        /*
         * clears dGamma and dBeta allocations by model.
         * @param modelID model this value belongs to.
         */
        void clear_dBeta_And_dGamma_By_Model(const std::string& modelID);

        // ===================== Forward/backward activation caching =====================

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_Input(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_Z(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_ActivationOutput(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_DAct(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_Delta(const std::string& modelID, const std::string& layerID, size_t length);

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getZ(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(zByModel_, "zByModel_", modelID, layerID);
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getActivationOutput(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(activationOutputsByModel_, "activationOutputsByModel_", modelID, layerID);
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getDAct(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(dActByModel_, "dActByModel_", modelID, layerID);
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getDelta(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(deltasByModel_, "deltasByModel_", modelID, layerID);
        }
        
        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getInput(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(inputsByModel_, "inputsByModel_", modelID, layerID);
        }

        /**
         * used to create buffer for dGamma
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_dGamma(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * used to create buffer for dBeta
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreate_dBeta(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * use to create resuable OpenCL buffer. If a buffer exist having such global and local reference ID, it will return that buffer rather creating a fresh buffer.
         * @param globalReferenceID global reference ID
         * @param localReferenceID local reference ID
         * @param length requested element count; the cached allocation grows as needed
         */
        cl_mem getOrCreateOpenCLBuffer(const std::string& globalReferenceID, const std::string& localReferenceID, size_t length);

        /**
         * exclusive only for layer norm's operation for gradient accumulation for gamma gradients
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem get_dGamma(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(dGammaByModel_, "dGammaByModel_", modelID, layerID);
        }

        /**
         * exclusive only for layer norm's operation for gradient accumulation for beta gradients
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem get_dBeta(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(dBetaByModel_, "dBetaByModel_", modelID, layerID);
        }

        cl_context context() { 
            return context_; 
        }
        cl_command_queue queue() { 
            return queue_; 
        }

        cl_kernel kernel(const std::string& name) const;

    private:
        GpuContext() = default;
        ~GpuContext() { 
            shutdown(); 
        }
        GpuContext(const GpuContext&) = delete;
        GpuContext& operator=(const GpuContext&) = delete;

        bool has_gpu_ = false;
        cl_platform_id platform_ = nullptr;
        cl_device_id   device_   = nullptr;
        cl_context     context_  = nullptr;
        cl_command_queue queue_  = nullptr;
        cl_program     program_  = nullptr;
        BufferCache weightsByModel_;
        BufferCache biasesByModel_;
        std::unordered_map<std::string, cl_kernel> kernels_;

        // ===================== Forward/backward activation caching =====================

        // Same caching idea as weights/biases above, but for the
        // per-layer values produced during feedforward/backprop: z (matmul output),
        // activation_output (post-activation), dAct (raw activation derivative), and
        // delta (dAct * incoming delta). Caching these on the GPU means gradient
        // accumulation can look them up by (modelID, layerID) instead of re-uploading
        // them from JS every call — they never leave the GPU between the layer op that
        // produces them and the gradient accumulation call that consumes them.
        BufferCache zByModel_;
        BufferCache activationOutputsByModel_;
        BufferCache dActByModel_;
        BufferCache deltasByModel_;
        BufferCache dGammaByModel_;
        BufferCache dBetaByModel_;
        BufferCache inputsByModel_;
        BufferCache somethingToWriteOn_;

        // Shared allocator for kernel-written buffers, growing existing cache entries as needed.
        cl_mem getOrCreateCacheBuffer(BufferCache& store, const std::string& modelID, const std::string& layerID, size_t length);
};