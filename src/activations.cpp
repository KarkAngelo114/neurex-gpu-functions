/*
    This is the native code of activations and their derivatives
    - Relu
    - Sigmoid
    - Tanh
    - Softmax
    - Linear

    - ReLu Derivative
    - Sigmoid Derivative
    - Tanh Derivative
    - Linear Derivative
    - Softmax Derivative

    All functions returns a 1D array

*/

#include <napi.h>
#include <CL/cl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "gpu/gpu_context.h"
#include "globals/globals.h"


/* ========================= Callable functions ============================*/

Napi::Value Relu_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("relu");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);


    return input;

}

Napi::Value Relu_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
    return input;
}

Napi::Value ReluWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Relu_GPU(info);
    }
    return Relu_CPU(info);
}

Napi::Value Sigmoid_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("sigmoid");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);

    return input;
}

Napi::Value Sigmoid_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();

    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = 1.0f / (1.0f + exp(-data[i]));
    }

    return input;
}

Napi::Value SigmoidWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Sigmoid_GPU(info);
    }
    return Sigmoid_CPU(info);
}

Napi::Value Tanh_GPU(const Napi::CallbackInfo& info) { 
    
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("Tanh");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);

    return input;
}

Napi::Value Tanh_CPU(const Napi::CallbackInfo& info) { 

   Napi::Float32Array input = info[0].As<Napi::Float32Array>();

   float* data = input.Data();
   size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = tanh(data[i]);
    }

   return input;
}

Napi::Value TanhWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Tanh_GPU(info);
    }

    return Tanh_CPU(info);
}

Napi::Value Softmax_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();

    float* data = input.Data();
    int inputSize = input.ElementLength();

    float max_val = data[0];
    for (int i = 0; i < inputSize; i++) max_val = std::max(max_val, data[i]);

    float sum = 0.0f;
    for (int i = 0; i < inputSize; i++) sum += std::exp(data[i] - max_val);

    auto& gpu = GpuContext::instance();

    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("softmax");

    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, data, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(float), &max_val);
    clSetKernelArg(kernel, 2, sizeof(float), &sum);
    clSetKernelArg(kernel, 3, sizeof(int), &inputSize);

    size_t globalSize = (size_t)inputSize;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(float) * inputSize, data, 0, nullptr, nullptr);

    clFinish(queue);

    clReleaseMemObject(buffer);

    return input;
}

Napi::Value Softmax_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    float max_val = data[0];
    for (size_t i = 1; i < input_size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input_size; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    for (size_t i = 0; i < input_size; i++) {
        data[i] /= sum;
    }

    return input;
}

Napi::Value SoftmaxWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Softmax_GPU(info);
    }

    return Softmax_CPU(info);
}

Napi::Value LinearWrapper(const Napi::CallbackInfo& info) {
    return info[0];
}

/* ========================= Derivatives ============================*/

Napi::Value DReLu_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("drelu");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);

    return input;
}

Napi::Value DReLu_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
    }
    return input;
}

Napi::Value DReLuWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DReLu_GPU(info);
    }

    return DReLu_CPU(info);
}

Napi::Value DSigmoid_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dsigmoid");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);

    return input;
}

Napi::Value DSigmoid_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        float s = 1.0f / (1.0f + std::exp(-data[i]));
        data[i] = s * (1.0f - s);
    }
    return input;
}

Napi::Value DSigmoidWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DSigmoid_GPU(info);
    }

    return DSigmoid_CPU(info);
}

Napi::Value DTanh_GPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dtanh");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, inputData, CL_TRUE, 0, sizeof(float)* input_size, input.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(inputData);
    
    return input;
}

Napi::Value DTanh_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();

    for (size_t i = 0; i < input_size; i++) {
        float t = std::tanh(data[i]);
        data[i] = 1.0f - (t * t);
    }
    return input;
}

Napi::Value DTanhWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DTanh_GPU(info);
    }

    return DTanh_CPU(info);
}

Napi::Value DSoftmaxWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int arr_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_size);

    std::fill(output.Data(), output.Data() + arr_size, 1.0f);

    return output;
}

Napi::Value DLinearWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t arr_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_size);

    std::fill(output.Data(), output.Data() + arr_size, 1.0f);

    return output;
}

void ActivationsRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("Relu", Napi::Function::New(env, ReluWrapper));
    exports.Set("Sigmoid", Napi::Function::New(env, SigmoidWrapper));
    exports.Set("Tanh", Napi::Function::New(env, TanhWrapper));
    exports.Set("Softmax", Napi::Function::New(env, SoftmaxWrapper));
    exports.Set("Linear", Napi::Function::New(env, LinearWrapper));
    exports.Set("DReLu", Napi::Function::New(env, DReLuWrapper));
    exports.Set("DSigmoid", Napi::Function::New(env, DSigmoidWrapper));
    exports.Set("DTanh", Napi::Function::New(env, DTanhWrapper));
    exports.Set("DSoftmax", Napi::Function::New(env, DSoftmaxWrapper));
    exports.Set("DLinear", Napi::Function::New(env, DLinearWrapper));
}