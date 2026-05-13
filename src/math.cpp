#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>

Napi::Value element_wise_mul_GPU(const CallbackInfo& info) {
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

    size_t globalSize = arr_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);
    clEnqueueReadBuffer(queue, ouptut_arr, CL_TRUE, 0, sizeof(float) * arr_length, output.Data(), 0, nullptr, nullptr);

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
    int arr_length = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    for (size_t i = 0; i < arr_length; i++) {
        o[i] = a1[i] * a2[i];
    }

    return output;
}

Napi::Value element_wise_mul_wrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (get_Global_Boolean_On_GPU()) {
        return element_wise_mul_GPU(info);
    }

    return element_wise_mul_CPU(info);
}

void Math_OPS(Napi::Env env, Napi::Object exports) {
    exports.Set("element_wise_mul", Napi::Function::New(env, element_wise_mul_wrapper));
}
