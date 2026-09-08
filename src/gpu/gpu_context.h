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

        bool initialize(const std::string& kernelBasePath, std::string& errorOut);
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

        // shared helper: fetch-or-create a cached buffer inside one of the maps above
        cl_mem getOrCreateStateBuffer(std::unordered_map<std::string, CL_MEM_ARRAY>& store, const std::string& modelID, int pointer, size_t length, const float* initialData);
};