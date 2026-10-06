#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <variant>
using FloatArray = std::vector<float>;
using Matrix = std::vector<FloatArray>;
using CL_MEM_ARRAY = std::vector<cl_mem>;
using BufferKey = std::variant<std::string, int>;
using BufferCache = std::unordered_map<std::string, std::map<BufferKey, cl_mem>>;

inline std::string describeBufferKey(const BufferKey& key) {
    if (const auto* layerID = std::get_if<std::string>(&key)) {
        return "layerID='" + *layerID + "'";
    }
    return "pointer=" + std::to_string(std::get<int>(key));
}

inline void validateBufferKey(const BufferKey& key) {
    if (const auto* layerID = std::get_if<std::string>(&key)) {
        if (layerID->empty()) {
            throw std::invalid_argument("GpuContext: layerID must not be empty.");
        }
    } else if (std::get<int>(key) < 0) {
        throw std::invalid_argument("GpuContext: cache pointer must not be negative.");
    }
}

// Looks up a keyed cache entry and reports its cache, model, and key on failure.
inline cl_mem lookupCachedBuffer(const BufferCache& store, const std::string& cacheName, const std::string& modelID, const BufferKey& key) {
    validateBufferKey(key);
    auto modelIt = store.find(modelID);
    if (modelIt == store.end()) {
        throw std::out_of_range("GpuContext: no '" + cacheName + "' cache entries for modelID='" + modelID + "' (" + describeBufferKey(key) + ").");
    }
    auto bufferIt = modelIt->second.find(key);
    if (bufferIt == modelIt->second.end()) {
        throw std::out_of_range("GpuContext: '" + cacheName + "' has no buffer for modelID='" + modelID + "' (" + describeBufferKey(key) + ").");
    }
    return bufferIt->second;
}

inline cl_mem lookupCachedBuffer(const std::unordered_map<std::string, CL_MEM_ARRAY>& store, const std::string& cacheName, const std::string& modelID, int pointer) {
    if (pointer < 0) {
        throw std::invalid_argument("GpuContext: " + cacheName + " pointer must not be negative.");
    }
    auto modelIt = store.find(modelID);
    if (modelIt == store.end()) {
        throw std::out_of_range("GpuContext: no '" + cacheName + "' cache entries for modelID='" + modelID + "'.");
    }
    const size_t index = static_cast<size_t>(pointer);
    if (index >= modelIt->second.size()) {
        throw std::out_of_range("GpuContext: '" + cacheName + "' has " + std::to_string(modelIt->second.size()) +
            " entries for modelID='" + modelID + "', but pointer=" + std::to_string(pointer) + " was requested.");
    }
    return modelIt->second[index];
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
        bool uploadParams(const std::string& modelID, const Matrix& weightMatrix, const Matrix& biasMatrix, std::string& errorOut);

        // release one model's buffers
        void clearParams(const std::string& modelID);

        // release everything (used by shutdown())
        void clearAllParams();

        /**
         * fetches the corresponding weights for the current layer, scoped to modelID
         * @param modelID use to reference what model's weights will get
         * @param pointer use to reference the specific layer's parameter
         * @return a clBuffer
         */
        cl_mem getWeights(const std::string& modelID, int pointer) const {
            return lookupCachedBuffer(weightsByModel_, "weightsByModel_", modelID, pointer);
        }

        /**
         * fetches the corresponding biases for the current layer, scoped to modelID
         * @param modelID use to reference what model's weights will get
         * @param pointer use to reference the specific layer's parameter
         * @return a clBuffer
         */
        cl_mem getBiases(const std::string& modelID, int pointer) const {
            return lookupCachedBuffer(biasesByModel_, "biasesByModel_", modelID, pointer);
        }

        // ===================== Optimizer state caching =====================
        //
        // Same idea as weights/biases caching above: instead of creating a fresh
        // clBuffer for m/v/velocity/sqAvg on every single optimizer step, we cache
        // one persistent clBuffer per (modelID, pointer, weights-or-biases) and just
        // update it in place. "getOrCreate" lazily allocates on first touch (seeded
        // with the host-side initial state, usually all zeros) and simply returns
        // the cached buffer on every call after that.

        /**
         * fetches (or lazily creates) the cached Adam "m" state buffer, scoped to modelID + pointer + param type
         * @param modelID model this state belongs to
         * @param pointer layer pointer within the model
         * @param isWeights true = weight-state map, false = bias-state map
         * @param length element count, used only the first time the buffer is created
         * @param initialData host data to seed the buffer with the first time it's created (may be nullptr for zero-init)
         * @return a persistent clBuffer holding the m state
         */
        cl_mem getOrCreate_M(const std::string& modelID, int pointer, bool isWeights, size_t length, const float* initialData);

        /** same as getOrCreate_M but for Adam's "v" (second moment) state */
        cl_mem getOrCreate_V(const std::string& modelID, int pointer, bool isWeights, size_t length, const float* initialData);

        /** same as getOrCreate_M but for SGD's "velocity" state */
        cl_mem getOrCreate_Velocity(const std::string& modelID, int pointer, bool isWeights, size_t length, const float* initialData);

        /** same as getOrCreate_M but for RMSProp's "sqAvg" (squared average) state */
        cl_mem getOrCreate_SqAvg(const std::string& modelID, int pointer, bool isWeights, size_t length, const float* initialData);

        // release every cached optimizer-state buffer for one model (called from clearParams)
        void clearOptimizerStates(const std::string& modelID);

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
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_Z(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_ActivationOutput(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_DAct(const std::string& modelID, const std::string& layerID, size_t length);

        /**
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_Delta(const std::string& modelID, const std::string& layerID, size_t length);

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getZ(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(zByModel_, "zByModel_", modelID, BufferKey{layerID});
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getActivationOutput(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(activationOutputsByModel_, "activationOutputsByModel_", modelID, BufferKey{layerID});
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getDAct(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(dActByModel_, "dActByModel_", modelID, BufferKey{layerID});
        }

        /** Read-only lookups
         * @param modelID model this value belongs to
         * @param layerID an identicator for a layer output
         */
        cl_mem getDelta(const std::string& modelID, const std::string& layerID) const {
            return lookupCachedBuffer(deltasByModel_, "deltasByModel_", modelID, BufferKey{layerID});
        }

        /**
         * used to create buffer for dGamma
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created 
         */
        cl_mem getOrCreate_dGamma(const std::string& modelID, int pointer, size_t length);

        /**
         * used to create buffer for dBeta
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created 
         */
        cl_mem getOrCreate_dBeta(const std::string& modelID, int pointer, size_t length);

        /**
         * exclusive only for layer norm's operation for gradient accumulation for gamma gradients
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem get_dGamma(const std::string& modelID, int pointer) const {
            return lookupCachedBuffer(dGammaByModel_, "dGammaByModel_", modelID, BufferKey{pointer});
        }

        /**
         * exclusive only for layer norm's operation for gradient accumulation for beta gradients
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem get_dBeta(const std::string& modelID, int pointer) const {
            return lookupCachedBuffer(dBetaByModel_, "dBetaByModel_", modelID, BufferKey{pointer});
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
        std::unordered_map<std::string, CL_MEM_ARRAY> weightsByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> biasesByModel_;
        std::unordered_map<std::string, cl_kernel> kernels_;

        // Optimizer state caches, keyed by modelID -> per-layer clBuffer, split by
        // weights vs biases just like weightsByModel_ / biasesByModel_ above.
        std::unordered_map<std::string, CL_MEM_ARRAY> mStatesWeights_;
        std::unordered_map<std::string, CL_MEM_ARRAY> mStatesBiases_;
        std::unordered_map<std::string, CL_MEM_ARRAY> vStatesWeights_;
        std::unordered_map<std::string, CL_MEM_ARRAY> vStatesBiases_;
        std::unordered_map<std::string, CL_MEM_ARRAY> velocityWeights_;
        std::unordered_map<std::string, CL_MEM_ARRAY> velocityBiases_;
        std::unordered_map<std::string, CL_MEM_ARRAY> sqAvgWeights_;
        std::unordered_map<std::string, CL_MEM_ARRAY> sqAvgBiases_;

        // ===================== Forward/backward activation caching =====================

        // Same caching idea as weights/biases and optimizer states above, but for the
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

        // shared helper: fetch-or-create a cached buffer inside one of the maps above
        cl_mem getOrCreateStateBuffer(std::unordered_map<std::string, CL_MEM_ARRAY>& store, const std::string& modelID, int pointer, size_t length, const float* initialData);

        // Shared lazy allocator for caches keyed by either layerID or integer key.
        // These buffers are written by kernels, so they do not need seed data.
        cl_mem getOrCreateCacheBuffer(BufferCache& store, const std::string& modelID, const BufferKey& key, size_t length);
};