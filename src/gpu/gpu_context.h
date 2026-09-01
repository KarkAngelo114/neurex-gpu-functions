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

        // Called from JS (once) after detectGPU() decided we have one.
        bool initialize(const std::string& kernelBasePath, std::string& errorOut);
        bool shutdown();

        bool hasGPU() { 
            return has_gpu_; 
        }

        bool uploadParams(const Matrix& weightMatrix, const Matrix& biasMatrix, std::string& errorOut);

        void clearParams();

        /**
         * fetches the corresponding weights for the current layer
         * @param pointer
         * @return cl_mem of weights
         */
        cl_mem getWeights(int pointer) {
            return weights[pointer];
        }

        /**
         * fetches the corresponding biases for the current layer
         * @param pointer
         * @return cl_mem of biases
         */
        cl_mem getBiases(int pointer) {
            return biases[pointer];
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
        CL_MEM_ARRAY weights;
        CL_MEM_ARRAY biases;
        std::unordered_map<std::string, cl_kernel> kernels_;
};