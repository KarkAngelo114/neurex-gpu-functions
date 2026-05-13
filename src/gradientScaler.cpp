#include <napi.h>
#include <omp.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"


Napi::Value ScaleGrads_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env(); // 1. Define env
    Napi::Float32Array grads = info[0].As<Napi::Float32Array>();
    int batchSize = info[1].As<Napi::Number>().Int32Value();

    cl_int err;
    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("scaleGrads");

    // 2. Create buffer
    cl_mem inputGrads = clCreateBuffer(gpu.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * grads.ElementLength(), grads.Data(), &err);
    
    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    // 3. Set Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputGrads);
    clSetKernelArg(kernel, 1, sizeof(int), &batchSize);

    // 4. Execute
    size_t global = grads.ElementLength();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // 5. Read back into a NEW Float32Array
    Napi::Float32Array scaledGrads = Napi::Float32Array::New(env, grads.ElementLength());
    clEnqueueReadBuffer(queue, inputGrads, CL_TRUE, 0, sizeof(float) * grads.ElementLength(), scaledGrads.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputGrads);
    
    return scaledGrads; // 6. Return the correct object
}

Napi::Value ScaleGrads_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array grads = info[0].As<Napi::Float32Array>();
    int batchSize = info[1].As<Napi::Number>().Int32Value();

    float* data = grads.Data();
    size_t length = grads.ElementLength();

    for (size_t i = 0; i < length; i++) {
        data[i] /= batchSize;
    }

    return grads;
}

Napi::Value ScaleGradientsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ScaleGrads_GPU(info);
    }

    return ScaleGrads_CPU(info);
    
}


void ScaleGradients(const Napi::Env env, const Napi::Object exports) {
   exports.Set("scaleGrad", Napi::Function::New(env, ScaleGradientsWrapper));
}