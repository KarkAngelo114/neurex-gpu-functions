#include <napi.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <CL/cl.h>
#include <cmath>

Napi::Value GradientClipping_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array grads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    int size = grads.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("gradientClipping");

    cl_mem accumulatedGrads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* size, grads.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &accumulatedGrads);
    clSetKernelArg(kernel, 1, sizeof(float), &threshold);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    size_t globalSize = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, accumulatedGrads, CL_TRUE, 0, sizeof(float)* size, grads.Data(), 0, nullptr, nullptr);
    clReleaseMemObject(accumulatedGrads);

    return grads;

}

Napi::Value GradientClipping_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputGrads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    size_t size = inputGrads.ElementLength();

    float* grads = inputGrads.Data();

    float norm = 0.0f;
    for (size_t i = 0; i < size; i++) {
        norm += grads[i] * grads[i];
    }

    norm = std::sqrt(norm);

    if (norm > threshold) {
        float scalingVal = threshold / norm;
        for (size_t i = 0; i < size; i++) {
            grads[i] *= scalingVal;
        }
    }

    return inputGrads;
}

Napi::Value gradientClippingWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return GradientClipping_GPU(info);
    }
    return GradientClipping_CPU(info);
}


void normalizers(Napi::Env env, Napi::Object exports) {
    exports.Set("gradientClipping", Napi::Function::New(env, gradientClippingWrapper));
}