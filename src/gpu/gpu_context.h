#pragma once
#include <CL/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
using Array = std::vector<float>;
using Matrix = std::vector<Array>;

class GpuContext {
    public:
        static GpuContext& instance();

        // Called from JS (once) after detectGPU() decided we have one.
        bool initialize(const std::string& kernelBasePath, std::string& errorOut);
        void shutdown();

        bool hasGPU() { 
            return has_gpu_; 
        }

        bool uploadParams(
            const Matrix& weights,
            const Matrix& biases,
            const Matrix& outputs,
            std::string& errorOut
        );


        /**
         * fetches the corresponding weight paramater using a pointer
         * @param pointer
         * @return a Float32Array of weights
         */
        cl_mem weight(int pointer) { 
            return d_weights_[pointer]; 
        }

        /**
         * fetches the corresponding biases paramater using a pointer
         * @param pointer
         * @return a Float32Array of biases
         */
        cl_mem bias(int pointer) { 
            return d_biases_[pointer]; 
        }

        /**
         * fetches the corresponding output tensor template for the current layer (for feedfoward process only) using a pointer
         * @param pointer
         * @return a Float32Array of output tensor template
         */
        cl_mem output(int pointer) { 
            return d_outputs_[pointer]; 
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
        std::vector<cl_mem> d_weights_;
        std::vector<cl_mem> d_biases_;
        std::vector<cl_mem> d_outputs_;
        std::unordered_map<std::string, cl_kernel> kernels_;
};