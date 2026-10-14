#pragma once
#include <vector>
#include <string>
#include <napi.h>
using Array = std::vector<float>;

std::string getComputeBackendType();

/**
 * get or create a modifiable Napi::Float32Array buffer. Note: The returning buffer indexed by `global_reference_id` and `local_reference_id` might contained old data. If you ever use this buffer for accumulation operations, consider zeroing it out first, otherwise, you get garbage data.
 * @param env Node env
 * @param global_reference_id a string value used to reference to which global store group the buffer belongs
 * @param local_reference_id a string value used to reference the buffer itself
 */
Napi::Float32Array getOrCreateBuffer(Napi::Env env, const std::string& global_reference_id, const std::string& local_reference_id, int size);
