#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>
#include <cmath>

Napi::Value SGD_GPU(const Napi::CallbackInfo& info) {
     Napi::Env env = info.Env();

    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array velocity = info[2].As<Napi::Float32Array>();
    float lr = info[3].As<Napi::Number>().FloatValue();
    float momentum = info[4].As<Napi::Number>().FloatValue();

    int param_length = params.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("sgd");

    cl_mem parameters = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* param_length, params.Data(), nullptr);
    cl_mem gradients = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* param_length, grads.Data(), nullptr);
    cl_mem velocity_array = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* param_length, velocity.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameters);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradients);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &velocity_array);
    clSetKernelArg(kernel, 3, sizeof(float), &lr);
    clSetKernelArg(kernel, 4, sizeof(float), &momentum);
    clSetKernelArg(kernel, 5, sizeof(int), &param_length);

    size_t globalSize = (size_t)param_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, parameters, CL_TRUE, 0, sizeof(float)* param_length, params.Data(), 0, nullptr, nullptr );
    clEnqueueReadBuffer(queue, velocity_array, CL_TRUE, 0, sizeof(float)* param_length, velocity.Data(), 0, nullptr, nullptr );

    clReleaseMemObject(parameters);
    clReleaseMemObject(gradients);
    clReleaseMemObject(velocity_array);

    Napi::Object output = Napi::Object::New(env);
    output.Set("params", params);
    output.Set("velocity", velocity);

    return output;
}

Napi::Value SGD_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array velocity = info[2].As<Napi::Float32Array>();
    float lr = info[3].As<Napi::Number>().FloatValue();
    float momentum = info[4].As<Napi::Number>().FloatValue();

    size_t element_length = params.ElementLength();

    float* p = params.Data();
    float* g = grads.Data();
    float* v = velocity.Data();

    #pragma omp parallel for
    for (size_t i = 0; i < element_length; i++) {
        v[i] = momentum * v[i] + g[i];
        p[i] -= lr * v[i];
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("params", params);
    output.Set("velocity", velocity);

    return output;
}

Napi::Value Adam_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array stateM = info[2].As<Napi::Float32Array>();
    Napi::Float32Array stateV = info[3].As<Napi::Float32Array>();
    int stateT = info[4].As<Napi::Number>().Int32Value();
    float learning_rate = info[5].As<Napi::Number>().FloatValue();
    float beta1 = info[6].As<Napi::Number>().FloatValue();
    float beta2 = info[7].As<Napi::Number>().FloatValue();
    float epsilon = info[8].As<Napi::Number>().FloatValue();
    int params_len = params.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("adam");

    cl_mem parameters = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* params_len, params.Data(), nullptr);
    cl_mem gradients = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* params_len, grads.Data(), nullptr);
    cl_mem M = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* params_len, stateM.Data(), nullptr);
    cl_mem V = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* params_len, stateV.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameters);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradients);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &M);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V);
    clSetKernelArg(kernel, 4, sizeof(int), &stateT);
    clSetKernelArg(kernel, 5, sizeof(float), &learning_rate);
    clSetKernelArg(kernel, 6, sizeof(float), &beta1);
    clSetKernelArg(kernel, 7, sizeof(float), &beta2);
    clSetKernelArg(kernel, 8, sizeof(float), &epsilon);
    clSetKernelArg(kernel, 9, sizeof(float), &params_len);

    size_t globalSize = (size_t)params_len;
    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &globalSize, nullptr, 0, nullptr, nullptr);
    
    // reads the paramters, M, and V buffers
    clEnqueueReadBuffer(queue, parameters, CL_TRUE, 0, sizeof(float)* params_len, params.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, M, CL_TRUE, 0, sizeof(float)* params_len, stateM.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, V, CL_TRUE, 0, sizeof(float)* params_len, stateV.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(parameters);
    clReleaseMemObject(gradients);
    clReleaseMemObject(M);
    clReleaseMemObject(V);

    Napi::Object output = Napi::Object::New(env);
    output.Set("params", params);
    output.Set("m", stateM);
    output.Set("v", stateV);

    return output;
}

Napi::Value Adam_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array params = info[0].As<Napi::Float32Array>();
    Napi::Float32Array grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array stateM = info[2].As<Napi::Float32Array>();
    Napi::Float32Array stateV = info[3].As<Napi::Float32Array>();
    int stateT = info[4].As<Napi::Number>().Int32Value();
    float learning_rate = info[5].As<Napi::Number>().FloatValue();
    float beta1 = info[6].As<Napi::Number>().FloatValue();
    float beta2 = info[7].As<Napi::Number>().FloatValue();
    float epsilon = info[8].As<Napi::Number>().FloatValue();
    int params_len = params.ElementLength();

    float* p = params.Data();
    float* g = grads.Data();
    float* sm = stateM.Data();
    float* sv = stateV.Data();

    #pragma omp parallel for
    for (int i = 0; i < params_len; i++) {
        float grad = g[i];

        sm[i] = beta1 * sm[i] + (1 - beta1) * grad;
        sv[i] = beta2 * sv[i] + (1 - beta2) * grad * grad;

        float mHat = sm[i] / (1.0f - pow(beta1, stateT));
        float vHat = sv[i] / (1.0f - pow(beta2, stateT));

        p[i] -= learning_rate * mHat / (sqrt(vHat) + epsilon);
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("params", params);
    output.Set("m", stateM);
    output.Set("v", stateV);

    return output;
}

Napi::Value SGD_wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return SGD_GPU(info);
    }

    return SGD_CPU(info);
}

Napi::Value Adam_wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Adam_GPU(info);
    }

    return Adam_CPU(info);
}

// ======== exports ======== //
void OptimizerInternals(Napi::Env env, Napi::Object exports) {
    exports.Set("SGD", Napi::Function::New(env, SGD_wrapper));
    exports.Set("Adam", Napi::Function::New(env, Adam_wrapper));
}