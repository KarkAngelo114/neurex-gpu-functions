#include <napi.h>
#include <omp.h>
#include "functions/functions.h"
#include "gpu/gpu_context.h"
#include "globals/globals.h"

void _helper_matMul(
    const float* input, 
    int inputSize, 
    int outputSize, 
    const float* weights, 
    const float* biases, 
    float* output
) {
    #pragma omp unroll partial(4)
    for (int j = 0; j < outputSize; j++) {
        output[j] = biases[j];
    }

    
    for (int i = 0; i < inputSize; i++) {
        float inputVal = input[i];
        int rowStart = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            output[j] += inputVal * weights[rowStart + j];
        }
    }
}

Napi::Value projectToQKV_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array Q_weights_tensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Q_bias_tensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array K_weights_tensor = info[3].As<Napi::Float32Array>();
    Napi::Float32Array K_bias_tensor = info[4].As<Napi::Float32Array>();
    Napi::Float32Array V_weights_tensor = info[5].As<Napi::Float32Array>();
    Napi::Float32Array V_bias_tensor = info[6].As<Napi::Float32Array>();
    int embeddingDim = info[7].As<Napi::Number>().Int32Value();
    int sequenceLen = info[8].As<Napi::Number>().Int32Value();
    int size = embeddingDim * sequenceLen;

    Napi::Float32Array Q = Napi::Float32Array::New(env, size);
    Napi::Float32Array K = Napi::Float32Array::New(env, size);
    Napi::Float32Array V = Napi::Float32Array::New(env, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("projectQKV");


    Napi::Float32Array concat_weights = concatenateFloat32Array(env, {Q_weights_tensor, K_weights_tensor, V_weights_tensor});
    Napi::Float32Array concat_bias = concatenateFloat32Array(env, {Q_bias_tensor, K_bias_tensor, V_bias_tensor});


    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* inputTensor.ElementLength(), inputTensor.Data(), nullptr);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* concat_weights.ElementLength(), concat_weights.Data(), nullptr);
    cl_mem biases = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* concat_bias.ElementLength(), concat_bias.Data(), nullptr);
    cl_mem _Q = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);
    cl_mem _K = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);
    cl_mem _V = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_Q);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &_K);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &_V);
    clSetKernelArg(kernel, 6, sizeof(int), &embeddingDim);
    clSetKernelArg(kernel, 7, sizeof(int), &sequenceLen);

    size_t global = (size_t)sequenceLen;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer( queue, _Q, CL_TRUE, 0, sizeof(float) * size, Q.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer( queue, _K, CL_TRUE, 0, sizeof(float) * size, K.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer( queue, _V, CL_TRUE, 0, sizeof(float) * size, V.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input);
    clReleaseMemObject(weights);
    clReleaseMemObject(biases);
    clReleaseMemObject(_Q);
    clReleaseMemObject(_K);
    clReleaseMemObject(_V);

    Napi::Object output = Napi::Object::New(env);
    output.Set("Q", Q);
    output.Set("K", K);
    output.Set("V", V);

    return output;
}

Napi::Value projectToQKV_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array Q_weights_tensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Q_bias_tensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array K_weights_tensor = info[3].As<Napi::Float32Array>();
    Napi::Float32Array K_bias_tensor = info[4].As<Napi::Float32Array>();
    Napi::Float32Array V_weights_tensor = info[5].As<Napi::Float32Array>();
    Napi::Float32Array V_bias_tensor = info[6].As<Napi::Float32Array>();
    int embeddingDim = info[7].As<Napi::Number>().Int32Value();
    int sequenceLen = info[8].As<Napi::Number>().Int32Value();
    int size = embeddingDim * sequenceLen;

    Napi::Float32Array Q = Napi::Float32Array::New(env, size);
    Napi::Float32Array K = Napi::Float32Array::New(env, size);
    Napi::Float32Array V = Napi::Float32Array::New(env, size);

    // Extract raw pointers for C++ CPU operations
    const float* inputPtr = inputTensor.Data();
    const float* Q_w = Q_weights_tensor.Data();
    const float* Q_b = Q_bias_tensor.Data();
    const float* K_w = K_weights_tensor.Data();
    const float* K_b = K_bias_tensor.Data();
    const float* V_w = V_weights_tensor.Data();
    const float* V_b = V_bias_tensor.Data();

    float* Q_ptr = Q.Data();
    float* K_ptr = K.Data();
    float* V_ptr = V.Data();

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < sequenceLen; t++) {
        int offset = t * embeddingDim;

        const float* tokenVec = inputPtr + offset;

        // Perform linear projection onto Q, K, and V
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, Q_w, Q_b, Q_ptr + offset);
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, K_w, K_b, K_ptr + offset);
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, V_w, V_b, V_ptr + offset);
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("Q", Q);
    output.Set("K", K);
    output.Set("V", V);

    return output;
}


Napi::Value projectToQKV_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return projectToQKV_GPU(info);
    }
    return projectToQKV_CPU(info);
}


void attentionFunctions(Napi::Env env, Napi::Object exports) {
    exports.Set("projectToQKV", Napi::Function::New(env, projectToQKV_Wrapper));
}