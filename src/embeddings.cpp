#include <napi.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <vector>
using IntArray = std::vector<int>;
using FloatArray= std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray Vector;
    Vector.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        Vector.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return Vector;
}

Napi::Value GetEmbeddings_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    IntArray tokenArray = Vectorize(info[0].As<Napi::Array>());
    int embeddingDim = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array params = info[2].As<Napi::Float32Array>();

    int sequence_length = tokenArray.size();
    int totalSize = sequence_length * embeddingDim; // token array length * embeddingDim = output size

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("getEmbeddings");

    cl_mem tokenBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * sequence_length, tokenArray.data(), nullptr);
    cl_mem lookup = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* params.ElementLength(), params.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* totalSize, nullptr, nullptr); 

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &tokenBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &lookup);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &embeddingDim);
    clSetKernelArg(kernel, 4, sizeof(int), &sequence_length);

    size_t globalSize[2] = {
        (size_t)sequence_length,
        (size_t)embeddingDim
    };
    clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalSize, nullptr, 0, nullptr, nullptr);

    // Read result back from GPU output buffer
    Napi::Float32Array result = Napi::Float32Array::New(env, totalSize);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * totalSize, result.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(output);
    clReleaseMemObject(tokenBuffer);
    clReleaseMemObject(lookup);


    return result;

}

Napi::Value GetEmbeddings_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    IntArray tokenArray = Vectorize(info[0].As<Napi::Array>());
    int embeddingDim = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array params = info[2].As<Napi::Float32Array>();
    int sequenceLength = tokenArray.size();
    int totalSize = sequenceLength * embeddingDim;

    Napi::Float32Array outputVector = Napi::Float32Array::New(env, totalSize);

    float* lookup = params.Data();
    float* output = outputVector.Data();

    // Iterate through each token in the sequence
    for (int i = 0; i < sequenceLength; i++) {
        int tokenID = tokenArray[i];
        int startIdx = tokenID * embeddingDim;
        
        // Copy the embedding row into the output at the correct offset
        for (int j = 0; j < embeddingDim; j++) {
            output[i * embeddingDim + j] = lookup[startIdx + j];
        }
    }


    return outputVector;
}


Napi::Value ReturnEmbeddings_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    IntArray activation_outputs = Vectorize(info[0].As<Napi::Array>());
    Napi::Float32Array delta = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int embeddingDim = info[3].As<Napi::Number>().Int32Value();
    int sequence_length = activation_outputs.size();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("returnEmbeddings");

    cl_mem tokenBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)* sequence_length, activation_outputs.data(), nullptr);
    cl_mem delta_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* delta.ElementLength(), delta.Data(), nullptr);
    cl_mem gradients = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)* weightGrads.ElementLength(), weightGrads.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &tokenBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradients);
    clSetKernelArg(kernel, 3, sizeof(int), &embeddingDim);
    clSetKernelArg(kernel, 4, sizeof(int), &sequence_length);

    size_t globalSize[2] = {
        (size_t)sequence_length,
        (size_t)embeddingDim
    };

    clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalSize, nullptr, 0, nullptr,nullptr);
    clEnqueueReadBuffer(queue, gradients, CL_TRUE, 0, sizeof(float)* weightGrads.ElementLength(), weightGrads.Data(), 0, nullptr, nullptr);
    clFinish(queue);
    clReleaseMemObject(tokenBuffer);
    clReleaseMemObject(delta_input);
    clReleaseMemObject(gradients);

    return weightGrads;
}

Napi::Value ReturnEmbeddings_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    IntArray activation_outputs = Vectorize(info[0].As<Napi::Array>());
    Napi::Float32Array delta = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int embeddingDim = info[3].As<Napi::Number>().Int32Value();

    float* deltaData = delta.Data();
    float* gradsData = weightGrads.Data();

    for (size_t i = 0; i < activation_outputs.size(); i++) {
        int tokenId = activation_outputs[i];

        if (tokenId == 0) continue;  // skip PAD tokens (ID 0)

        int gradOffset = tokenId * embeddingDim;
        int deltaOffset = i * embeddingDim;

        for (int d = 0; d < embeddingDim; d++) {
            gradsData[gradOffset + d] += deltaData[deltaOffset + d];
        }
    }

    return weightGrads;
}



Napi::Value GetEmbeddingsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return GetEmbeddings_GPU(info);
    }
    return GetEmbeddings_CPU(info);
}

Napi::Value ReturnEmbeddingsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ReturnEmbeddings_GPU(info);
    }
    return ReturnEmbeddings_CPU(info);
}

void Embeddings(const Napi::Env env, const Napi::Object exports) {
    exports.Set("getEmbeddings", Napi::Function::New(env, GetEmbeddingsWrapper));
    exports.Set("returnEmbeddings", Napi::Function::New(env, ReturnEmbeddingsWrapper));
}