#include <napi.h>
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

Napi::Value MSE_GPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("mse");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);
    
    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float sum = 0;
    for (size_t i = 0; i < occurrence; i++) {
        sum += results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return sum / occurrence;
}

Napi::Value MSE_CPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    int occurrence = preds.ElementLength();
    float sum = 0;

    float* p = preds.Data();
    float* a = acts.Data();

    for (int i = 0; i < occurrence; i++) {
        float difference = p[i] - a[i];
        sum += difference * difference;
    }

    return sum / occurrence;

}

Napi::Value MAE_GPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("mae");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, acts.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * occurrence, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predictions);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &actuals);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &occurrence);
    
    size_t globalSize = (size_t)occurrence;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    FloatArray results(occurrence);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * occurrence, results.data(), 0, nullptr, nullptr);

    float sum = 0;
    for (size_t i = 0; i < occurrence; i++) {
        sum += results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return sum / occurrence;


}

Napi::Value MAE_CPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    int occurrence = preds.ElementLength();
    float sum = 0;

    float* p = preds.Data();
    float* a = acts.Data();

    for (int i = 0; i < occurrence; i++) {
        sum += std::abs(p[i] - a[i]);
    }

    return sum / occurrence;
}

Napi::Value CCE_GPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("cce");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, acts.Data(), nullptr);
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

    float loss = 0;
    for (size_t i = 0; i < occurrence; i++) {
        loss -= results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return loss;
}

Napi::Value CCE_CPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    auto epsilon = info[2].As<Napi::Number>().DoubleValue();
    int length = preds.ElementLength();
    float loss = 0;

    float* p = preds.Data();
    float* a = acts.Data();

    for (int i = 0; i < length; i++) {
        loss -= a[i] * std::log(std::max(p[i], epsilon));
    }

    return loss;

}

Napi::Value SCCE_CPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    IntArray actual_label = Vectorize(info[1].As<Napi::Array>());
    auto epsilon = info[2].As<Napi::Number>().DoubleValue();
    int length = preds.ElementLength();

    float* p = preds.Data();

    float max = std::max(p[actual_label[0]], epsilon);

    return -std::log(max);
    
}

Napi::Value BCE_GPU(Napi::CallbackInfo info) {
    Napi::Float32Array preds = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    size_t occurrence = preds.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("bce");

    cl_mem predictions = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, preds.Data(), nullptr);
    cl_mem actuals = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* occurrence, acts.Data(), nullptr);
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

    float sum = 0;
    for (size_t i = 0; i < occurrence; i++) {
        sum -= results[i];
    }

    clReleaseMemObject(predictions);
    clReleaseMemObject(actuals);
    clReleaseMemObject(output);

    return sum / occurrence;
}

Napi::Value BCE_CPU(Napi::CallbackInfo info) {
    Napi::Float32Array predictions = info[0].As<Napi::Float32Array>();
    Napi::Float32Array acts = info[1].As<Napi::Float32Array>();
    auto epsilon = info[2].As<Napi::Number>().DoubleValue();
    int length = predictions.ElementLength();
    float sum = 0;

    float* preds = predictions.Data();
    float* a = acts.Data();

    for (int i = 0; i < length; i++) {
        float p = std::max(std::min(preds[i], 1 - epsilon), epsilon);
        sum -= a[i] * std::log(p) + (1 - a[i]) * std::log(1 - p);
    }

    return sum / length;

}

// =========================== wrappers ========================= //
Napi::Value MSE_Wrapper(Napi::CallbackInfo info) {
    if (get_Global_Boolean_On_GPU()) {
        return MSE_GPU(info);
    }

    return MSE_CPU(info);
}

Napi::Value MAE_Wrapper(Napi::CallbackInfo info) {
    if (get_Global_Boolean_On_GPU()) {
        return MAE_GPU(info);
    }

    return MAE_CPU(info);
}

Napi::Value CCE_Wrapper(Napi::CallbackInfo info) {
    if (get_Global_Boolean_On_GPU()) {
        return CCE_GPU(info);
    }
    return CCE_CPU(info);
}

Napi::Value SCCE_Wrapper(Napi::CallbackInfo info) {
    return SCCE_CPU(info); // doesn't need GPU branch as we can use the labels as index to access the value from the predictions
}

Napi::Value BCE_Wrapper(Napi::CallbackInfo info) {
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