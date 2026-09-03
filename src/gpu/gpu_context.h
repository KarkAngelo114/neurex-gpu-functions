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
};