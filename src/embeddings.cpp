#include <napi.h>
#include <omp.h>
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
    int pointer = info[3].As<Napi::Number>().Int32Value();
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();

    int sequence_length = tokenArray.size();
    int totalSize = sequence_length * embeddingDim; // token array length * embeddingDim = output size

    Napi::Float32Array result = getOrCreateBuffer(env, modelID, layerID+"_embeddings", totalSize);

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("getEmbeddings");

    cl_mem tokenBuffer = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(totalSize));
    clEnqueueWriteBuffer(queue, tokenBuffer, CL_FALSE, 0, sizeof(int) * sequence_length, tokenArray.data(), 0, nullptr, nullptr);
    cl_mem lookup = gpu.getWeights(modelID, pointer);
    cl_mem output = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_forward_embeddings", static_cast<size_t>(totalSize));

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
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * totalSize, result.Data(), 0, nullptr, nullptr);

    return result;

}

Napi::Value GetEmbeddings_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    IntArray tokenArray = Vectorize(info[0].As<Napi::Array>());
    int embeddingDim = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array params = info[2].As<Napi::Float32Array>();
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();
    int sequenceLength = tokenArray.size();
    int totalSize = sequenceLength * embeddingDim;

    Napi::Float32Array outputVector = getOrCreateBuffer(env, modelID, layerID+"_embeddings", totalSize);

    float* lookup = params.Data();
    float* output = outputVector.Data();

    // Iterate through each token in the sequence
    for (int i = 0; i < sequenceLength; i++) {
        int tokenID = tokenArray[i];
        int startIdx = tokenID * embeddingDim;
        
        
        // Copy the embedding row into the output at the correct offset
        #pragma omp unroll partial(4)
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
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();

    int sequence_length = activation_outputs.size();
    int deltaSize = delta.ElementLength();
    int weightGradSize = weightGrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("returnEmbeddings");

    cl_mem tokenBuffer = gpu.getInput(modelID, layerID);
    cl_mem delta_input = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_embeddings_incoming_delta", static_cast<size_t>(deltaSize));
    clEnqueueWriteBuffer(queue, delta_input, CL_FALSE, 0, sizeof(float) * deltaSize, delta.Data(), 0, nullptr, nullptr);
    cl_mem gradients = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_embeddings_gradients", static_cast<size_t>(weightGradSize));
    clEnqueueWriteBuffer(queue, gradients, CL_FALSE, 0, sizeof(float) * weightGradSize, weightGrads.Data(), 0, nullptr, nullptr);

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
    clEnqueueReadBuffer(queue, gradients, CL_TRUE, 0, sizeof(float)* weightGradSize, weightGrads.Data(), 0, nullptr, nullptr);


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
    
    for (int i = 0; i < (int)activation_outputs.size(); i++) {
        int tokenId = activation_outputs[i];

        if (tokenId == 0) continue;  // skip PAD tokens (ID 0)

        int gradOffset = tokenId * embeddingDim;
        int deltaOffset = i * embeddingDim;

        #pragma omp unroll partial(4)        
        for (int d = 0; d < embeddingDim; d++) {
            gradsData[gradOffset + d] += deltaData[deltaOffset + d];
        }
    }

    return weightGrads;
}

Napi::Value GetEmbeddingsWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return GetEmbeddings_GPU(info);
    }
    return GetEmbeddings_CPU(info);
}

Napi::Value ReturnEmbeddingsWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return ReturnEmbeddings_GPU(info);
    }
    return ReturnEmbeddings_CPU(info);
}

void Embeddings(const Napi::Env env, const Napi::Object exports) {
    exports.Set("getEmbeddings", Napi::Function::New(env, GetEmbeddingsWrapper));
    exports.Set("returnEmbeddings", Napi::Function::New(env, ReturnEmbeddingsWrapper));
}