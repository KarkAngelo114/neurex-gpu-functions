#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
using FloatArray = std::vector<float>;
using Matrix = std::vector<FloatArray>;
using CL_MEM_ARRAY = std::vector<cl_mem>;


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
            return weightsByModel_.at(modelID).at(static_cast<size_t>(pointer));
        }

        /**
         * fetches the corresponding biases for the current layer, scoped to modelID
         * @param modelID use to reference what model's weights will get
         * @param pointer use to reference the specific layer's parameter
         * @return a clBuffer
         */
        cl_mem getBiases(const std::string& modelID, int pointer) const {
            return biasesByModel_.at(modelID).at(static_cast<size_t>(pointer));
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
         * fetches (or lazily creates) the cached buffer for this layer's z (matmul output),
         * writable by the producer and readable by consumers (activation
         * kernels, delta derivative kernels).
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_Z(const std::string& modelID, int pointer, size_t length);

        /**
         * creates layer's final activation output (post-activation) cached for gradient accumulation
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_ActivationOutput(const std::string& modelID, int pointer, size_t length);

        /**
         * creates layer's derivative activation output cached for getting the final delta of a layer. (example: dAct * incoming delta via element-wise-multiplication)
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created
         */
        cl_mem getOrCreate_DAct(const std::string& modelID, int pointer, size_t length);


        /**
         * Used to create buffer for final delta for this layer. Often used by layers that has operator after derivative activation. (example: dAct * incoming delta via element-wise-multiplication)
         * Once this function is called, `getDelta` can be use to look it up using a pointer and modelID, usually used for gradient accumulation step. Note: you can only use this buffer allocator for one final output delta per layer.
         * Attempting to use this to another final output delta per layer on the same pointer will get overwritten (like the case of layer norm where its deltas to be use for accumulation needs 2 buffes to cache).
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         * @param length element count, used only the first time the buffer is created 
         */
        cl_mem getOrCreate_Delta(const std::string& modelID, int pointer, size_t length);

        /** Read-only lookups for consumers that expect the value to already be cached.
         * Throws (via .at()) if nothing was cached yet for this (modelID, pointer) — that means something ran out of order, which should fail loudly, not silently.
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem getZ(const std::string& modelID, int pointer) const {
            return zByModel_.at(modelID).at(static_cast<size_t>(pointer));
        }

        /**
         * used in any operator that requires cached post-activated output (usually created by `getOrCreate_ActivationOutput`)
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem getActivationOutput(const std::string& modelID, int pointer) const {
            return activationOutputsByModel_.at(modelID).at(static_cast<size_t>(pointer));
        }

        /**
         * used in any operator that requires cached derivative activation output (usually created by `getOrCreate_DAct`)
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem getDAct(const std::string& modelID, int pointer) const {
            return dActByModel_.at(modelID).at(static_cast<size_t>(pointer));
        }

        /**
         * used in any operator that requires cached final delta output per layer (usually created by `getOrCreate_Delta`)
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem getDelta(const std::string& modelID, int pointer) const {
            return deltasByModel_.at(modelID).at(static_cast<size_t>(pointer));
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
            return dGammaByModel_.at(modelID).at(static_cast<size_t>(pointer));
        }

        /**
         * exclusive only for layer norm's operation for gradient accumulation for beta gradients
         * @param modelID model this value belongs to
         * @param pointer layer pointer within the model
         */
        cl_mem get_dBeta(const std::string& modelID, int pointer) const {
            return dBetaByModel_.at(modelID).at(static_cast<size_t>(pointer));
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
        // accumulation can look them up by (modelID, pointer) instead of re-uploading
        // them from JS every call — they never leave the GPU between the layer op that
        // produces them and the gradient accumulation call that consumes them.
        std::unordered_map<std::string, CL_MEM_ARRAY> zByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> activationOutputsByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> dActByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> deltasByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> dGammaByModel_;
        std::unordered_map<std::string, CL_MEM_ARRAY> dBetaByModel_;

        // shared helper: fetch-or-create a cached buffer inside one of the maps above
        cl_mem getOrCreateStateBuffer(std::unordered_map<std::string, CL_MEM_ARRAY>& store, const std::string& modelID, int pointer, size_t length, const float* initialData);

        // shared helper for the forward/backward caches above: same lazy-alloc idea,
        // but with no seed data — the first write to a slot is always a kernel
        // computing the real value, so there's nothing worth zero-initializing.
        cl_mem getOrCreateCacheBuffer(std::unordered_map<std::string, CL_MEM_ARRAY>& store, const std::string& modelID, int pointer, size_t length);
};