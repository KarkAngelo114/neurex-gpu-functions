#include <napi.h>
#include <CL/cl.h>
#include <omp.h>
#include <cmath>
#include <limits>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;


static IntArray Vectorize(const Napi::Array& arr) {
    IntArray Vector;
    Vector.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        Vector.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return Vector;
}

Napi::Value MaxPooling_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    cl_int err;

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    IntArray pool_size = Vectorize(info[1].As<Napi::Array>());
    IntArray inputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[3].As<Napi::Array>());
    size_t strides = info[4].As<Napi::Number>().Int32Value();

    int poolH = pool_size[0];
    int poolW = pool_size[1];

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int inputD = inputShape[2];

    int outputH = outputShape[0];
    int outputW = outputShape[1];
    int outputD = outputShape[2];

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("maxpool");

    size_t inputSize = inputH * inputW * inputD;
    size_t outputSize = outputH * outputW * outputD;

    // INPUT BUFFER
    cl_mem inputTensor = clCreateBuffer(gpu.context(),CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input_array.Data(),&err);

    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create input buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    // OUTPUT BUFFER
    cl_mem outputTensor = clCreateBuffer(gpu.context(), CL_MEM_WRITE_ONLY, sizeof(float) * outputSize, nullptr, &err);

    // MAX INDEX BUFFER
    cl_mem maxIndexTensor = clCreateBuffer(gpu.context(), CL_MEM_WRITE_ONLY, sizeof(int) * outputSize, nullptr, &err);

    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create output buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    // SET KERNEL ARGS (FIXED POINTER USAGE)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputTensor);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &maxIndexTensor);

    clSetKernelArg(kernel, 3, sizeof(int), &inputH);
    clSetKernelArg(kernel, 4, sizeof(int), &inputW);
    clSetKernelArg(kernel, 5, sizeof(int), &inputD);
    clSetKernelArg(kernel, 6, sizeof(int), &poolH);
    clSetKernelArg(kernel, 7, sizeof(int), &poolW);
    clSetKernelArg(kernel, 8, sizeof(int), &outputH);
    clSetKernelArg(kernel, 9, sizeof(int), &outputW);
    clSetKernelArg(kernel, 10, sizeof(int), &outputD);
    clSetKernelArg(kernel, 11, sizeof(int), &strides);

    size_t globalSize[3] = {
        (size_t)outputH,
        (size_t)outputW,
        (size_t)outputD
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // READ BACK RESULTS
    std::vector<float> output(outputSize);
    std::vector<int> maxIdx(outputSize);

    clEnqueueReadBuffer(queue, outputTensor, CL_TRUE, 0, sizeof(float) * outputSize, output.data(), 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, maxIndexTensor, CL_TRUE, 0, sizeof(int) * outputSize, maxIdx.data(), 0, nullptr, nullptr);

    // CLEANUP
    clReleaseMemObject(inputTensor);
    clReleaseMemObject(outputTensor);
    clReleaseMemObject(maxIndexTensor);

    // BUILD JS OUTPUT
    Napi::Float32Array outArray = Napi::Float32Array::New(env, outputSize);
    Napi::Int32Array maxArray = Napi::Int32Array::New(env, outputSize);

    memcpy(outArray.Data(), output.data(), sizeof(float) * outputSize);
    memcpy(maxArray.Data(), maxIdx.data(), sizeof(int) * outputSize);

    Napi::Object objectOutput = Napi::Object::New(env);
    objectOutput.Set("output", outArray);
    objectOutput.Set("maxIndices", maxArray);

    return objectOutput;
}

Napi::Value MaxPooling_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    IntArray pool_size = Vectorize(info[1].As<Napi::Array>());
    IntArray inputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[3].As<Napi::Array>());
    size_t strides = info[4].As<Napi::Number>().Int32Value();
    
    size_t poolH = pool_size[0];
    size_t poolW = pool_size[1];

    size_t inputH = inputShape[0];
    size_t inputW = inputShape[1];
    size_t inputD = inputShape[2];

    size_t outputH = outputShape[0];
    size_t outputW = outputShape[1];
    size_t outputD = outputShape[2];

    // prepare output
    Napi::Float32Array output = Napi::Float32Array::New(env, outputH * outputW * outputD);
    Napi::Int32Array maxArray = Napi::Int32Array::New(env, outputH * outputW * outputD);

    float* arr = input_array.Data();
    int* max = maxArray.Data();
    float* out = output.Data();

    for (size_t d = 0; d < inputD; d++) {
        for (size_t i = 0; i < outputH; i++) {
            for (size_t j = 0;  j < outputW; j++) {
                float maxVal = -std::numeric_limits<float>::infinity();
                int maxIdx = -1;

                size_t startH = i * strides;
                size_t startW = j * strides;

                for (size_t ph = 0; ph < poolH; ph++) {
                    for (size_t pw = 0; pw < poolW; pw++) {
                        int currH = startH + ph;
                        int currW = startW + pw;

                        // Check bounds to handle cases where window might exceed input dimensions
                        if (currH < inputH && currW < inputW) {
                            // Calculate index in the flattened 1D array
                            int idx = (currH * inputW * inputD) + (currW * inputD) + d;
                            float val = arr[idx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }
                }
                int outIdx = (i * outputW * outputD) + (j * outputD) + d;
                out[outIdx] = (maxVal == -std::numeric_limits<float>::infinity()) ? 0.0f : maxVal;
                max[outIdx] = maxIdx;
                
            }
        }
    }

    Napi::Object objectOutput= Napi::Object::New(env);
    objectOutput.Set("output", output);
    objectOutput.Set("maxIndices", maxArray);

    return objectOutput;
}

Napi::Value MaxPoolDelta_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    Napi::Int32Array indicesArray = info[1].As<Napi::Int32Array>();
    int H = info[2].As<Napi::Number>().Int32Value();
    int W = info[3].As<Napi::Number>().Int32Value();
    int D = info[4].As<Napi::Number>().Int32Value();
    int size = H * W * D;

    Napi::Float32Array output = Napi::Float32Array::New(env, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("maxpooldelta");

    cl_mem inputData = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_arr.ElementLength(), input_arr.Data(), nullptr);
    cl_mem indices = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, sizeof(int)* indicesArray.ElementLength(), indicesArray.Data(), nullptr);
    cl_mem outputTensor = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* size, output.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputData);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputTensor);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    size_t globalSize = indicesArray.ElementLength();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    // READ BACK RESULTS
    clEnqueueReadBuffer(queue, outputTensor, CL_TRUE, 0, sizeof(float) * size, output.Data(), 0, nullptr, nullptr);
    clFinish(queue);

    // CLEANUP
    clReleaseMemObject(inputData);
    clReleaseMemObject(indices);
    clReleaseMemObject(outputTensor);

    return output;
}

Napi::Value MaxPoolDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    Napi::Int32Array indicesArray = info[1].As<Napi::Int32Array>();
    int H = info[2].As<Napi::Number>().Int32Value();
    int W = info[3].As<Napi::Number>().Int32Value();
    int D = info[4].As<Napi::Number>().Int32Value();
    int size = H * W * D;

    Napi::Float32Array output = Napi::Float32Array::New(env, size);

    float* inputData = input_arr.Data();
    int* indices = indicesArray.Data();
    float* o = output.Data();

    for (size_t i = 0; i < indicesArray.ElementLength(); i++) {
        int idx = indices[i];
        o[idx] += inputData[i]; 
    }

    return output;
}

Napi::Value MaxPoolingWrapper(const Napi::CallbackInfo& info) {

    if (get_Global_Boolean_On_GPU()) {
        return MaxPooling_GPU(info);
    }

    return MaxPooling_CPU(info);
}

Napi::Value MaxPoolDelta_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return MaxPoolDelta_GPU(info);
    }

    return MaxPoolDelta_CPU(info);
}


void Poolings(Napi::Env env, Napi::Object exports) {
    exports.Set("MaxPooling", Napi::Function::New(env, MaxPoolingWrapper));
    exports.Set("MaxPoolDelta", Napi::Function::New(env, MaxPoolDelta_Wrapper));
}