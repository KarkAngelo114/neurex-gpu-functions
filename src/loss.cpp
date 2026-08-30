#include <napi.h>
#include <omp.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <cmath>
#include <vector>
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

Napi::Value MSE_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("mse");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);

    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float sum = 0.0f;
    for (size_t i = 0; i < occurrence; i++) {
        sum += results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return Napi::Number::New(env, sum / occurrence);
}

Napi::Value MSE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    int occurrence = preds.ElementLength();
    float sum = 0.0f;

    float* p = preds.Data();
    float* a = acts.Data();

    #pragma omp unroll partial(4)
    for (int i = 0; i < occurrence; i++) {
        float difference = p[i] - a[i];
        sum += difference * difference;
    }

    return Napi::Number::New(env, sum / occurrence);
}

Napi::Value MAE_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("mae");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);

    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float sum = 0.0f;

    #pragma omp unroll partial(4)
    for (size_t i = 0; i < occurrence; i++) {
        sum += results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return Napi::Number::New(env, sum / occurrence);
}

Napi::Value MAE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    int occurrence = preds.ElementLength();
    float sum = 0.0f;

    float* p = preds.Data();
    float* a = acts.Data();

    #pragma omp unroll partial(4)
    for (int i = 0; i < occurrence; i++) {
        sum += std::abs(p[i] - a[i]);
    }

    return Napi::Number::New(env, sum / occurrence);
}

Napi::Value CCE_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    float epsilon = (float)info[2].As<Napi::Number>().DoubleValue();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("cce");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);
    clSetKernelArg(kernel, 4, sizeof(float), &epsilon);

    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float loss = 0.0f;
    #pragma omp unroll partial(4)
    for (size_t i = 0; i < occurrence; i++) {
        loss -= results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return Napi::Number::New(env, loss);
}

Napi::Value CCE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    float epsilon = (float)info[2].As<Napi::Number>().DoubleValue();
    int length = preds.ElementLength();
    float loss = 0.0f;

    float* p = preds.Data();
    float* a = acts.Data();

    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        loss -= a[i] * std::log(std::max(p[i], epsilon));
    }

    return Napi::Number::New(env, loss);
}

Napi::Value SCCE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    IntArray actual_label = Vectorize(info[1].As<Napi::Array>());
    float epsilon = (float)info[2].As<Napi::Number>().DoubleValue();

    float* p = preds.Data();

    float val = std::max(p[actual_label[0]], epsilon);

    return Napi::Number::New(env, -std::log(val));
}

Napi::Value BCE_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    float epsilon = (float)info[2].As<Napi::Number>().DoubleValue();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("bce");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);
    clSetKernelArg(kernel, 4, sizeof(float), &epsilon);

    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float sum = 0.0f;
    #pragma omp unroll partial(4)
    for (size_t i = 0; i < occurrence; i++) {
        sum -= results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return Napi::Number::New(env, sum / occurrence);
}

Napi::Value BCE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array predictions = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    float epsilon = (float)info[2].As<Napi::Number>().DoubleValue();
    int length = predictions.ElementLength();
    float sum = 0.0f;

    float* preds = predictions.Data();
    float* a = acts.Data();

    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        float p = std::max(std::min(preds[i], 1.0f - epsilon), epsilon);
        sum -= a[i] * std::log(p) + (1.0f - a[i]) * std::log(1.0f - p);
    }

    return Napi::Number::New(env, sum / length);
}

// =========================== wrappers ========================= //
Napi::Value MSE_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return MSE_GPU(info);
    }

    return MSE_CPU(info);
}

Napi::Value MAE_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return MAE_GPU(info);
    }

    return MAE_CPU(info);
}

Napi::Value CCE_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return CCE_GPU(info);
    }
    return CCE_CPU(info);
}

Napi::Value SCCE_Wrapper(const Napi::CallbackInfo& info) {
    return SCCE_CPU(info);
}

Napi::Value BCE_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return BCE_GPU(info);
    }

    return BCE_CPU(info);
}


void loss(Napi::Env env, Napi::Object exports) {
    exports.Set("mse", Napi::Function::New(env, MSE_Wrapper));
    exports.Set("mae", Napi::Function::New(env, MAE_Wrapper));
    exports.Set("categorical_cross_entropy", Napi::Function::New(env, CCE_Wrapper));
    exports.Set("sparse_categorical_cross_entropy", Napi::Function::New(env, SCCE_Wrapper));
    exports.Set("binary_cross_entropy", Napi::Function::New(env, BCE_Wrapper));
}
