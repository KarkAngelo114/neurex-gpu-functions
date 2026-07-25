#pragma once
#include <napi.h>
#include <vector>



Napi::Float32Array subarray(const Napi::Env env, const Napi::Float32Array& float32Array, int begin, int end);
std::vector<const float*> ExtractFloat32ArrayPointers(const Napi::Array& jsArray);






