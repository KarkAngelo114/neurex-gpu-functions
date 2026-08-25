#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>

Napi::Value element_wise_mul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    int arr_length = arr1.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("element_wise_mul");

    cl_mem input_arr1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr1.Data(), nullptr);
    cl_mem input_arr2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr2.Data(), nullptr);
    cl_mem output_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * arr_length, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_arr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_arr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 3, sizeof(int), &arr_length);

    size_t globalSize = (size_t)arr_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * arr_length, output.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(input_arr1);
    clReleaseMemObject(input_arr2);
    clReleaseMemObject(output_arr);
    
    return output;
}

Napi::Value element_wise_mul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    size_t arr_length = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    for (size_t i = 0; i < arr_length; i++) {
        o[i] = a1[i] * a2[i];
    }

    return output;
}

Napi::Value element_wise_sub_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    int arr_length = arr1.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("element_wise_sub");

    cl_mem input_arr1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr1.Data(), nullptr);
    cl_mem input_arr2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr2.Data(), nullptr);
    cl_mem output_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * arr_length, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_arr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_arr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 3, sizeof(int), &arr_length);

    size_t globalSize = (size_t)arr_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * arr_length, output.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(input_arr1);
    clReleaseMemObject(input_arr2);
    clReleaseMemObject(output_arr);
    
    return output;
}

Napi::Value element_wise_sub_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    size_t arr_length = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    for (size_t i = 0; i < arr_length; i++) o[i] = a1[i] - a2[i]; 

    return output;
}

Napi::Value scaleDiff_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    Napi::Float32Array arr3 = info[2].As<Napi::Float32Array>();
    int arr_length = arr1.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("scale_diff");

    cl_mem input_arr1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr1.Data(), nullptr);
    cl_mem input_arr2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr2.Data(), nullptr);
    cl_mem input_arr3 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * arr_length, arr3.Data(), nullptr);
    cl_mem output_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * arr_length, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_arr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_arr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_arr3);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 4, sizeof(int), &arr_length);

    size_t globalSize = (size_t)arr_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * arr_length, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_arr1);
    clReleaseMemObject(input_arr2);
    clReleaseMemObject(input_arr3);
    clReleaseMemObject(output_arr);
    
    return output;
}

Napi::Value scaleDiff_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    Napi::Float32Array arr3 = info[2].As<Napi::Float32Array>();
    int arr_length = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);
    float scale = 2.0f / arr_length;

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* a3 = arr3.Data();
    float* o = output.Data();

    for (int i = 0; i < arr_length; i++) {
        o[i] = (a1[i] - a2[i]) * a3[i] * scale;
    }

    return output;
}

Napi::Value Scale_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env(); // 1. Define env
    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int size = inputArray.ElementLength();
    int scalingFactor = info[1].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("scale");
    cl_context context = gpu.context();

    // 2. Create buffer
    cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) *size, inputArray.Data(), nullptr);

    // 3. Set Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(int), &scalingFactor);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    // 4. Execute
    size_t global = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // 5. Read back into a NEW Float32Array
    Napi::Float32Array scaledOutput = Napi::Float32Array::New(env, size);
    clEnqueueReadBuffer(queue, input, CL_TRUE, 0, sizeof(float) * size, scaledOutput.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input);
    
    return inputArray;
}

Napi::Value Scale_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int scalingFactor = info[1].As<Napi::Number>().Int32Value();

    float* data = inputArray.Data();
    size_t length = inputArray.ElementLength();

    for (size_t i = 0; i < length; i++) {
        data[i] /= scalingFactor;
    }

    return inputArray;
}

Napi::Value element_wise_mul_wrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (get_Global_Boolean_On_GPU()) {
        return element_wise_mul_GPU(info);
    }

    return element_wise_mul_CPU(info);
}

Napi::Value element_wise_sub_wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return element_wise_sub_GPU(info);
    }

    return element_wise_sub_CPU(info);

}

Napi::Value scaleDiffWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return scaleDiff_GPU(info);
    }

    return scaleDiff_CPU(info);

}

Napi::Value ScalerWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Scale_GPU(info);
    }

    return Scale_CPU(info);
    
}

void Math_OPS(Napi::Env env, Napi::Object exports) {
    exports.Set("element_wise_mul", Napi::Function::New(env, element_wise_mul_wrapper));
    exports.Set("element_wise_sub", Napi::Function::New(env, element_wise_sub_wrapper));
    exports.Set("scaleDiff", Napi::Function::New(env, scaleDiffWrapper));
    exports.Set("scale", Napi::Function::New(env, ScalerWrapper));
}
