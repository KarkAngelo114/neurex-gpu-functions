#pragma once
#include <napi.h>
#include <vector>

using NapiArrayChunk = std::vector<Napi::Float32Array>;


Napi::Float32Array subarray(const Napi::Env env, const Napi::Float32Array& float32Array, int begin, int end);
std::vector<const float*> ExtractFloat32ArrayPointers(const Napi::Array& jsArray);
Napi::Float32Array concatenateFloat32Array(const Napi::Env env, const NapiArrayChunk& chunks);






