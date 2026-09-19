#include <napi.h>
#include <omp.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <cmath>
#include <algorithm>
#include <cstring>


Napi::Value SPE_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int embeddingDim = info[1].As<Napi::Number>().Int32Value();
    int sequenceLength = info[2].As<Napi::Number>().Int32Value();
    int size = static_cast<int>(inputArray.ElementLength());

    int expectedSize = embeddingDim * sequenceLength;

    if (size != expectedSize) {
        std::string msg =
            "Sinusoidal encoding error: Input data size mismatch. Input dimension: "
            + std::to_string(size)
            + " | Expected: "
            + std::to_string(expectedSize);

        Napi::RangeError::New(env, msg).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Float32Array outputArray = Napi::Float32Array::New(env, size);

    if (size == 0) {
        return outputArray;
    }

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("spe");

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inputArray.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(int), &embeddingDim);
    clSetKernelArg(kernel, 2, sizeof(int), &sequenceLength);
    clSetKernelArg(kernel, 3, sizeof(int), &size);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);

    size_t globalSize = static_cast<size_t>(size);
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * size, outputArray.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input);
    clReleaseMemObject(output);

    return outputArray;
}

Napi::Value SPE_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int embeddingDim = info[1].As<Napi::Number>().Int32Value();
    int sequenceLength = info[2].As<Napi::Number>().Int32Value();
    int size = inputArray.ElementLength();
    Napi::Float32Array outputArray = Napi::Float32Array::New(env, size);

    if (size != embeddingDim * sequenceLength) {
        std::string msg = "Sinusoidal encoding error: Input data size mismatch. Input dimension: "
                        + std::to_string(size)
                        + " | Expected: "
                        + std::to_string(embeddingDim * sequenceLength);

        Napi::RangeError::New(env, msg).ThrowAsJavaScriptException();
        return env.Undefined();
    }

    float* input = inputArray.Data();
    float* output = outputArray.Data();

    // Match the JS implementation: copy the incoming buffer first, then add positional values.
    std::memcpy(output, input, size * sizeof(float));

    for (int pos = 0; pos < sequenceLength; pos++) {
        int offset = pos * embeddingDim;

        int i = 0;

        // Unroll the embedding-dimension loop four times per iteration.
        for (; i <= embeddingDim - 4; i += 4) {
            int pairIndex = std::floor(i / 2.0f);
            float exponent = (2.0f * pairIndex) / static_cast<float>(embeddingDim);
            float angle = static_cast<float>(pos) / std::pow(10000.0f, exponent);

            output[offset + i] += (i % 2 == 0) ? std::sin(angle) : std::cos(angle);

            int pairIndex1 = std::floor((i + 1) / 2.0f);
            float exponent1 = (2.0f * pairIndex1) / static_cast<float>(embeddingDim);
            float angle1 = static_cast<float>(pos) / std::pow(10000.0f, exponent1);
            output[offset + i + 1] += (i % 2 == 1) ? std::sin(angle1) : std::cos(angle1);

            int pairIndex2 = std::floor((i + 2) / 2.0f);
            float exponent2 = (2.0f * pairIndex2) / static_cast<float>(embeddingDim);
            float angle2 = static_cast<float>(pos) / std::pow(10000.0f, exponent2);
            output[offset + i + 2] += (i % 2 == 0) ? std::sin(angle2) : std::cos(angle2);

            int pairIndex3 = std::floor((i + 3) / 2.0f);
            float exponent3 = (2.0f * pairIndex3) / static_cast<float>(embeddingDim);
            float angle3 = static_cast<float>(pos) / std::pow(10000.0f, exponent3);
            output[offset + i + 3] += (i % 2 == 1) ? std::sin(angle3) : std::cos(angle3);
        }

        for (; i < embeddingDim; i++) {
            int pairIndex = std::floor(i / 2.0f);
            float exponent = (2.0f * pairIndex) / static_cast<float>(embeddingDim);
            float angle = static_cast<float>(pos) / std::pow(10000.0f, exponent);

            output[offset + i] += (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }

    return outputArray;
}


Napi::Value SPE_wrapper(const Napi::CallbackInfo& info) {
    return SPE_CPU(info);
}

void encoders(Napi::Env env, Napi::Object exports) {
    exports.Set("SinusoidalPositionalEncoding", Napi::Function::New(env, SPE_wrapper));
}
