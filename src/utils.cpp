#include <napi.h>
#include <omp.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <vector>
using IntArray = std::vector<int>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray vectorArray;
    vectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        vectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return vectorArray;
}

Napi::Value DilateInput_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    IntArray shape = Vectorize(info[1].As<Napi::Array>());
    int stride = info[2].As<Napi::Number>().Int32Value();

    int H = shape[0];
    int W = shape[1];
    int C = shape[2];

    int dilatedH = (H - 1) * stride + 1;
    int dilatedW = (W - 1) * stride + 1;
    int dilatedSize = dilatedH * dilatedW * C;

    Napi::Float32Array dilatedOutput = Napi::Float32Array::New(env, dilatedSize);


    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("dilate");

    cl_mem inputTensor = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_arr.ElementLength(), input_arr.Data(), nullptr);
    cl_mem outputTensor = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* dilatedSize, dilatedOutput.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputTensor);
    clSetKernelArg(kernel, 2, sizeof(int), &H);
    clSetKernelArg(kernel, 3, sizeof(int), &W);
    clSetKernelArg(kernel, 4, sizeof(int), &C);
    clSetKernelArg(kernel, 5, sizeof(int), &stride);
    clSetKernelArg(kernel, 6, sizeof(int), &dilatedH);
    clSetKernelArg(kernel, 7, sizeof(int), &dilatedW);
    
    size_t globalSize[3] = {
        (size_t)C,
        (size_t)H,
        (size_t)W
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, 0, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, outputTensor, CL_TRUE, 0, sizeof(float)* dilatedSize, dilatedOutput.Data(), 0, nullptr, nullptr);
    clFinish(queue);

    clReleaseMemObject(inputTensor);
    clReleaseMemObject(outputTensor);

    Napi::Object output = Napi::Object::New(env);

    output.Set("data", dilatedOutput);
    output.Set("dilatedHeight", dilatedH);
    output.Set("dilatedWidth", dilatedW);

    return output;
}

Napi::Value DilateInput_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    IntArray shape = Vectorize(info[1].As<Napi::Array>());
    size_t stride = info[2].As<Napi::Number>().Int32Value();

    size_t H = shape[0];
    size_t W = shape[1];
    size_t C = shape[2];

    size_t dilatedH = (H - 1) * stride + 1;
    size_t dilatedW = (W - 1) * stride + 1;
    
    int dilatedSize = dilatedH * dilatedW * C;

    Napi::Float32Array dilatedOutput = Napi::Float32Array::New(env, dilatedSize);

    float* input = input_arr.Data();
    float* dilated = dilatedOutput.Data();

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                size_t srcIdx = (h * W + w) * C + c;
                size_t dilatedHIdx = h * stride;
                size_t dilatedWIdx = w * stride;
                size_t dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
                dilated[dstIdx] = input[srcIdx];
            }
        }
    }

    Napi::Object output = Napi::Object::New(env);

    output.Set("data", dilatedOutput);
    output.Set("dilatedHeight", dilatedH);
    output.Set("dilatedWidth", dilatedW);

    return output;
}

Napi::Value ApplyPadding_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    int inputH = info[1].As<Napi::Number>().Int32Value();
    int inputW = info[2].As<Napi::Number>().Int32Value();
    int channels = info[3].As<Napi::Number>().Int32Value();
    int padTop = info[4].As<Napi::Number>().Int32Value();
    int padBottom = info[5].As<Napi::Number>().Int32Value();
    int padLeft = info[6].As<Napi::Number>().Int32Value();
    int padRight = info[7].As<Napi::Number>().Int32Value();

    int newH = inputH + padTop + padBottom;
    int newW = inputW + padLeft + padRight;
    size_t inputSize = inputH * inputW * channels;
    size_t outputSize = newH * newW * channels;
    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("apply_padding");

    cl_mem inputTensor = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputSize, input_arr.Data(), nullptr);
    cl_mem outputTensor = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputSize, nullptr, nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputTensor);
    clSetKernelArg(kernel, 2, sizeof(int), &inputH);
    clSetKernelArg(kernel, 3, sizeof(int), &inputW);
    clSetKernelArg(kernel, 4, sizeof(int), &channels);
    clSetKernelArg(kernel, 5, sizeof(int), &padTop);
    clSetKernelArg(kernel, 6, sizeof(int), &padBottom);
    clSetKernelArg(kernel, 7, sizeof(int), &padLeft);
    clSetKernelArg(kernel, 8, sizeof(int), &padRight);
    clSetKernelArg(kernel, 9, sizeof(int), &newH);
    clSetKernelArg(kernel, 10, sizeof(int), &newW);

    // Execute kernel with 3D work dimensions
    size_t globalSize[3] = {
        (size_t)newH,
        (size_t)newW,
        (size_t)channels
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, outputTensor, CL_TRUE, 0, sizeof(float) * outputSize, output.Data(), 0, nullptr, nullptr);

    // Cleanup
    clReleaseMemObject(inputTensor);
    clReleaseMemObject(outputTensor);
    clFinish(queue);

    // Create output array
    Napi::Float32Array outputData = Napi::Float32Array::New(env, outputSize);
    memcpy(outputData.Data(), output.Data(), sizeof(float) * outputSize);

    // Create shape vector
    IntArray newShape = { newH, newW, channels };

    // Build return object
    Napi::Object data = Napi::Object::New(env);
    data.Set("data", outputData);

    // Convert std::vector<int> to Napi::Array
    Napi::Array shapeArray = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArray.Set(i, Napi::Number::New(env, newShape[i]));
    }

    data.Set("shape", shapeArray);

    return data;

}

Napi::Value ApplyPadding_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    size_t inputH = info[1].As<Napi::Number>().Int32Value();
    size_t inputW = info[2].As<Napi::Number>().Int32Value();
    size_t channels = info[3].As<Napi::Number>().Int32Value();
    size_t padTop = info[4].As<Napi::Number>().Int32Value();
    size_t padBottom = info[5].As<Napi::Number>().Int32Value();
    size_t padLeft = info[6].As<Napi::Number>().Int32Value();
    size_t padRight = info[7].As<Napi::Number>().Int32Value();

    size_t newH = inputH + padTop + padBottom;
    size_t newW = inputW + padLeft + padRight;
    Napi::Float32Array outputData = Napi::Float32Array::New(env, newH * newW * channels);

    float* input = input_arr.Data();
    float* output = outputData.Data();

    for (size_t i = 0; i < inputH; i++) {
        for (size_t j = 0; j < inputW; j++) {
            for (size_t c = 0; c < channels; c++) {
                size_t oldIdx = (i * inputW + j) * channels + c;
                size_t newIdx = ((i + padTop) * newW + (j + padLeft)) * channels + c;
                output[newIdx] = input[oldIdx];
            }
        }
    }

    // Create the vector as you intended (using curly braces for initialization)
    IntArray newShape = { static_cast<int>(newH), static_cast<int>(newW), static_cast<int>(channels) };

    Napi::Object data = Napi::Object::New(env);
    data.Set("data", outputData);

    // Convert std::vector<int> to Napi::Array
    Napi::Array shapeArray = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArray.Set(i, Napi::Number::New(env, newShape[i]));
    }

    data.Set("shape", shapeArray);

    return data;

}

Napi::Value ApplyPadding_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ApplyPadding_GPU(info);
    }

    return ApplyPadding_CPU(info);
}

Napi::Value DilateInputWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return DilateInput_GPU(info);
    }
    return DilateInput_CPU(info);
}

void utils(Napi::Env env, Napi::Object exports) {
    exports.Set("DilateInput", Napi::Function::New(env, DilateInputWrapper));
    exports.Set("ApplyPadding", Napi::Function::New(env, ApplyPadding_Wrapper));
}