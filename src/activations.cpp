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
#include <omp.h>
#include <CL/cl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "gpu/gpu_context.h"
#include "globals/globals.h"


/* ========================= Callable functions ============================*/

Napi::Value Relu_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>(); // this won't be use in this branch for buffer creation, but only for size referencing
    int pointer = info[1].As<Napi::Number>().Int32Value();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();

    int input_size = input.ElementLength();
    
    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("relu");

    cl_mem inputData = gpu.getZ(modelID, pointer); // we get the pre-activated outputs cached by index and modelID
    cl_mem output = gpu.getOrCreate_ActivationOutput(modelID, pointer, static_cast<size_t>(input_size)); // the final activated output, will be cached by pointer and modelID. This must be cached because it will be use by gradient accumulator operators.

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);


    size_t globalSize = static_cast<size_t>(input_size);
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* input_size, outputArray.Data(), 0, nullptr, nullptr);

    return outputArray;

}

Napi::Value Relu_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);

    float* data = input.Data();
    float* output = outputArray.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
        output[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
    return outputArray;
}

Napi::Value Sigmoid_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>().Int32Value();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    int input_size = input.ElementLength();
    

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("sigmoid");

    cl_mem inputData = gpu.getZ(modelID, pointer);
    cl_mem output = gpu.getOrCreate_ActivationOutput(modelID, pointer, static_cast<size_t>(input_size));

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);

    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* input_size, outputArray.Data(), 0, nullptr, nullptr);

    return outputArray;
}

Napi::Value Sigmoid_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);

    float* data = input.Data();
    float* output = outputArray.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
        output[i] = 1.0f / (1.0f + exp(-data[i]));
    }

    return outputArray;
}

Napi::Value Tanh_GPU(const Napi::CallbackInfo& info) { 
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>().Int32Value();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    int input_size = input.ElementLength();
    

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("Tanh");

    cl_mem inputData = gpu.getZ(modelID, pointer);
    cl_mem output = gpu.getOrCreate_ActivationOutput(modelID, pointer, static_cast<size_t>(input_size));

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);

    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* input_size, outputArray.Data(), 0, nullptr, nullptr);

    return outputArray;
}

Napi::Value Tanh_CPU(const Napi::CallbackInfo& info) { 
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();

    float* data = input.Data();
    int input_size = input.ElementLength();
    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);
    float* output = outputArray.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
           output[i] = tanh(data[i]);
    }

       return outputArray;
}

Napi::Value Softmax_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>().Int32Value();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    int inputSize = input.ElementLength();

    float* data = input.Data();

    float max_val = data[0];
    for (int i = 0; i < inputSize; i++) max_val = std::max(max_val, data[i]);

    float sum = 0.0f;
    for (int i = 0; i < inputSize; i++) sum += std::exp(data[i] - max_val);

    auto& gpu = GpuContext::instance();

    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("softmax");

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, data, nullptr);
    cl_mem output = gpu.getOrCreate_ActivationOutput(modelID, pointer, static_cast<size_t>(inputSize));;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(float), &max_val);
    clSetKernelArg(kernel, 3, sizeof(float), &sum);
    clSetKernelArg(kernel, 4, sizeof(int), &inputSize);

    size_t globalSize = (size_t)inputSize;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array outputArray = Napi::Float32Array::New(env, inputSize);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * inputSize, outputArray.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputBuffer);

    return outputArray;
}

Napi::Value Softmax_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    float* data = input.Data();
    size_t input_size = input.ElementLength();
    Napi::Float32Array outputArray = Napi::Float32Array::New(env, input_size);
    float* output = outputArray.Data();

    float max_val = data[0];
    #pragma omp unroll partial(4)
    for (size_t i = 1; i < input_size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    #pragma omp unroll partial(4)
    for (size_t i = 0; i < input_size; i++) {
        output[i] = std::exp(data[i] - max_val);
        sum += output[i];
    }

    #pragma omp unroll partial(4)
    for (size_t i = 0; i < input_size; i++) {
        output[i] /= sum;
    }

    return outputArray;
}

/* ========================= Derivatives ============================*/

Napi::Value DReLu_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();

    size_t input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("drelu");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);
    cl_mem outputData = gpu.getOrCreate_DAct(modelID, pointer, input_size);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputData);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);

    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, outputData, CL_TRUE, 0, sizeof(float)* input_size, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputData);

    return output;
}

Napi::Value DReLu_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    
    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    float* inData = input.Data();
    float* outData = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
        outData[i] = inData[i] > 0.0f ? 1.0f : 0.0f;
    }
    return output;
}

Napi::Value DSigmoid_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();

    size_t input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dsigmoid");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);
    cl_mem outputData = gpu.getOrCreate_DAct(modelID, pointer, input_size);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputData);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, outputData, CL_TRUE, 0, sizeof(float)* input_size, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputData);

    return output;
}

Napi::Value DSigmoid_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    
    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    float* inData = input.Data();
    float* outData = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
        float s = 1.0f / (1.0f + std::exp(-inData[i]));
        outData[i] = s * (1.0f - s);
    }
    return output;
}

Napi::Value DTanh_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int pointer = info[1].As<Napi::Number>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();

    size_t input_size = input.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dtanh");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_size, input.Data(), nullptr);
    cl_mem outputData = gpu.getOrCreate_DAct(modelID, pointer, input_size);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputData);
    clSetKernelArg(kernel, 2, sizeof(int), &input_size);


    size_t globalSize = (size_t)input_size;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    clEnqueueReadBuffer(queue, outputData, CL_TRUE, 0, sizeof(float)* input_size, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputData);
    
    return output;
}

Napi::Value DTanh_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    
    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    float* inData = input.Data();
    float* outData = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < input_size; i++) {
        float t = std::tanh(inData[i]);
        outData[i] = 1.0f - (t * t);
    }
    return output;
}

// =============== wrappers ====================
Napi::Value ReluWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Relu_GPU(info);
    }
    return Relu_CPU(info);
}

Napi::Value SigmoidWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Sigmoid_GPU(info);
    }
    return Sigmoid_CPU(info);
}

Napi::Value TanhWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Tanh_GPU(info);
    }

    return Tanh_CPU(info);
}

Napi::Value SoftmaxWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Softmax_GPU(info);
    }

    return Softmax_CPU(info);
}

Napi::Value LinearWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int input_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, input_size);
    std::copy(input.Data(), input.Data() + input_size, output.Data());
    return output;
}

Napi::Value DReLuWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DReLu_GPU(info);
    }

    return DReLu_CPU(info);
}

Napi::Value DSigmoidWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DSigmoid_GPU(info);
    }

    return DSigmoid_CPU(info);
}

Napi::Value DSoftmaxWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1_input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2_input = info[1].As<Napi::Float32Array>();
    int arr_size = arr1_input.ElementLength();

    Napi::Float32Array outputArr = Napi::Float32Array::New(env, arr_size);

    float* arr1 = arr1_input.Data();
    float* arr2 = arr2_input.Data();
    float* output = outputArr.Data();

    float dot_product = 0.0f;
    
    #pragma omp unroll partial(4)
    for (int i = 0; i < arr_size; i++) {
        dot_product += arr2[i] * arr1[i];
    }

    #pragma omp unroll partial(4)
    for (int i = 0; i < arr_size; i++) {
        output[i] = arr1[i] * (arr2[i] - dot_product);
    }

    return outputArr;
}

Napi::Value DLinearWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t arr_size = input.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_size);

    std::fill(output.Data(), output.Data() + arr_size, 1.0f);

    return output;
}

Napi::Value DTanhWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DTanh_GPU(info);
    }

    return DTanh_CPU(info);
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