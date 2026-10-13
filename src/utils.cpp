#include <napi.h>
#include <vector>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
using IntArray = std::vector<int>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray vectorArray;
    vectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        vectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return vectorArray;
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

    IntArray newShape = { static_cast<int>(newH), static_cast<int>(newW), static_cast<int>(channels) };

    Napi::Object data = Napi::Object::New(env);
    data.Set("data", outputData);

    Napi::Array shapeArray = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArray.Set(i, Napi::Number::New(env, newShape[i]));
    }

    data.Set("shape", shapeArray);
    return data;
}

Napi::Value ApplyPaddingWrapper(const Napi::CallbackInfo& info) {
    return ApplyPadding_CPU(info);
}

Napi::Value cacheOutputLayerDelta(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array hostOutputDeltaBuffer = info[0].As<Napi::Float32Array>();
    std::string modelID = info[1].As<Napi::String>().Utf8Value();
    std::string layerID = info[2].As<Napi::String>().Utf8Value();
    int size = hostOutputDeltaBuffer.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_mem outputDeltaBuffer = gpu.getOrCreate_Delta(modelID, layerID, static_cast<size_t>(size));

    clEnqueueWriteBuffer(queue, outputDeltaBuffer, CL_TRUE, 0, sizeof(float)* size, hostOutputDeltaBuffer.Data(), 0, nullptr, nullptr);


    return env.Undefined();
}

// ================ wrappers ==================
Napi::Value cacheOutputLayerDeltaWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return cacheOutputLayerDelta(info);
    }
    // CPU mode: nothing to cache, there's no GPU context to write into.
    return info.Env().Undefined();
}

void utils(Napi::Env env, Napi::Object exports) {
    exports.Set("ApplyPadding", Napi::Function::New(env, ApplyPaddingWrapper));
    exports.Set("cacheOutputLayerDelta", Napi::Function::New(env, cacheOutputLayerDeltaWrapper));
}