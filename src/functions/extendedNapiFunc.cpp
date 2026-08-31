#include <napi.h>
#include "functions.h"
#include <vector>

using NapiArrayChunk = std::vector<Napi::Float32Array>;

/**
 * Gets a new Float32Array view of the ArrayBuffer passed to this function, referencing the elements at begin, inclusive, up to end, exclusive.
 *
 * @param env Napi env variable
 * @param float32array float32Array buffer
 * @param begin The index of the beginning of the array.
 * @param end The index of the end of the array.
 */
Napi::Float32Array subarray(const Napi::Env env, const Napi::Float32Array& float32Array, int begin, int end) {
    size_t arrayLength = float32Array.ElementLength();

    // 1. Boundary checking
    if (begin < 0 || end < 0 || begin > end || begin > arrayLength || end > arrayLength) {
        std::string msg = "Invalid subarray range [" + std::to_string(begin) + ", " + std::to_string(end) + "] for array length " + std::to_string(arrayLength);
        Napi::RangeError::New(env, msg).ThrowAsJavaScriptException();
        return Napi::Float32Array(); // Returns empty/null Napi handle after throwing
    }

    // 2. Calculate offsets in bytes
    size_t elementSize = sizeof(float);
    size_t byteOffset = float32Array.ByteOffset() + (begin * elementSize);
    size_t length = end - begin;

    // 3. Create a view on the existing ArrayBuffer
    return Napi::Float32Array::New(env, length, float32Array.ArrayBuffer(), byteOffset);
}

Napi::Float32Array concatenateFloat32Array(const Napi::Env env, const NapiArrayChunk& chunks) {
    size_t totalLength = 0;
    for (const auto& chunk : chunks) {
        totalLength += chunk.ElementLength();
    }

    Napi::ArrayBuffer arrayBuffer = Napi::ArrayBuffer::New(env, totalLength * sizeof(float));
    Napi::Float32Array result = Napi::Float32Array::New(env, totalLength, arrayBuffer, 0);

    float* destPtr = result.Data();
    size_t offset = 0;

    for (const auto& chunk : chunks) {
        size_t chunkLen = chunk.ElementLength();
        if (chunkLen > 0) {
            std::memcpy(destPtr + offset, chunk.Data(), chunkLen * sizeof(float));
            offset += chunkLen;
        }
    }

    return result;
}

/**
 * Extracts the float32Array pointers from Napi::Array
 * 
 * @param jsArray an array containing Float32Array
 */
std::vector<const float*> ExtractFloat32ArrayPointers(const Napi::Array& jsArray) {
    uint32_t len = jsArray.Length();
    std::vector<const float*> pointers(len);
    for (uint32_t i = 0; i < len; ++i) {
        Napi::Value el = jsArray.Get(i);
        pointers[i] = el.IsTypedArray() ? el.As<Napi::Float32Array>().Data() : nullptr;
    }
    return pointers;
}