#pragma once
#include <CL/cl.h>
#include <string>
#include <unordered_map>

class GpuContext {
    public:
        static GpuContext& instance();

        // Called from JS (once) after detectGPU() decided we have one.
        bool initialize(std::string& errorOut);
        void shutdown();

        bool hasGPU() const { return has_gpu_; }

        cl_context       context() const { return context_; }
        cl_command_queue queue()   const { return queue_; }
        cl_kernel        kernel(const std::string& name) const;

    private:
        GpuContext() = default;
        ~GpuContext() { shutdown(); }
        GpuContext(const GpuContext&) = delete;
        GpuContext& operator=(const GpuContext&) = delete;

        bool has_gpu_ = false;
        cl_platform_id platform_ = nullptr;
        cl_device_id   device_   = nullptr;
        cl_context     context_  = nullptr;
        cl_command_queue queue_  = nullptr;
        cl_program     program_  = nullptr;
        std::unordered_map<std::string, cl_kernel> kernels_;
};