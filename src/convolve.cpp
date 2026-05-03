#include <napi.h>
#include <omp.h>
#include <vector>
#include <cstring>
#include <algorithm>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;


// ======================= UTILS ================================ //
static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

std::vector<float> Rotate_kernels(int F, int KH, int KW, int D, int pointer) {
    const auto& kernels_arr = getGlobalWeights(pointer);

    size_t kernel_length = kernels_arr.size();

    FloatArray outputData(kernel_length);

    const float* kernels = kernels_arr.data();
    float* rotated = outputData.data();

    for (size_t f = 0; f < F; f++) {
        for (size_t kh = 0; kh < KH; kh++) {
            for (size_t kw = 0; kw < KW; kw++) {
                for (size_t d = 0; d < D; d++) {
                    // Original Index
                    size_t oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    
                    // Rotated Index (Flip KH and KW)
                    size_t newKh = KH - 1 - kh;
                    size_t newKw = KW - 1 - kw;
                    size_t newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = kernels[oldIdx];
                }
            }
        }
    }

    return outputData;
}


// ==================== MAIN FUNCTIONS ======================= //
Napi::Value Convolve_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    int output_height = info[2].As<Napi::Number>().Int32Value();
    int output_width = info[3].As<Napi::Number>().Int32Value();
    int num_filters = info[4].As<Napi::Number>().Int32Value();
    int kernel_height = info[5].As<Napi::Number>().Int32Value();
    int kernel_width = info[6].As<Napi::Number>().Int32Value();
    int depth = info[7].As<Napi::Number>().Int32Value();
    int input_height = info[8].As<Napi::Number>().Int32Value();
    int input_width = info[9].As<Napi::Number>().Int32Value();
    int pointer = info[10].As<Napi::Number>().Int32Value();
    int outputPointer = info[11].As<Napi::Number>().Int32Value();

    cl_int err;

    auto& gpu = GpuContext::instance();
    cl_mem inputTensor = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_height * input_width * depth, input.Data(), &err);
    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create input buffer for Convolution").ThrowAsJavaScriptException();
        return env.Null();
    }
    cl_command_queue queue = gpu.queue();
    cl_mem weights = gpu.weight(pointer);
    cl_mem biases = gpu.bias(pointer);
    cl_mem output_tensor = gpu.output(outputPointer);

    cl_kernel kernel = gpu.kernel("convolve");

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_tensor);
    clSetKernelArg(kernel, 4, sizeof(int), &strides);
    clSetKernelArg(kernel, 5, sizeof(int), &output_height);
    clSetKernelArg(kernel, 6, sizeof(int), &output_width);
    clSetKernelArg(kernel, 7, sizeof(int), &num_filters);
    clSetKernelArg(kernel, 8, sizeof(int), &kernel_height);
    clSetKernelArg(kernel, 9, sizeof(int), &kernel_width);
    clSetKernelArg(kernel, 10, sizeof(int), &depth);
    clSetKernelArg(kernel, 11, sizeof(int), &input_height);
    clSetKernelArg(kernel, 12, sizeof(int), &input_width);

    size_t global = output_height * output_width * num_filters;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    int expected_size = output_height * output_width * num_filters;
    Napi::Float32Array output = Napi::Float32Array::New(env, expected_size);

    clEnqueueReadBuffer(queue, output_tensor, CL_TRUE, 0, sizeof(float) * expected_size, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputTensor);

    return output;
}


Napi::Value Convolve_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    size_t strides = info[1].As<Napi::Number>().Int32Value();
    size_t output_height = info[2].As<Napi::Number>().Int32Value();
    size_t output_width = info[3].As<Napi::Number>().Int32Value();
    size_t num_filters = info[4].As<Napi::Number>().Int32Value();
    size_t kernel_height = info[5].As<Napi::Number>().Int32Value();
    size_t kernel_width = info[6].As<Napi::Number>().Int32Value();
    size_t depth = info[7].As<Napi::Number>().Int32Value();
    size_t input_height = info[8].As<Napi::Number>().Int32Value();
    size_t input_width = info[9].As<Napi::Number>().Int32Value();
    size_t pointer = info[10].As<Napi::Number>().Int32Value();
    size_t outputPointer = info[11].As<Napi::Number>().Int32Value();

    const auto& kernels_array = getGlobalWeights(pointer);
    const auto& biases_array = getGlobalBiases(pointer);

    float* data = input.Data();
    const float* kernels = kernels_array.data();
    const float* biases = biases_array.data();

    Napi::Float32Array output = Napi::Float32Array::New(env, output_height * output_width * num_filters);
    float* outData = output.Data();

    #pragma omp for schedule(static)
    for (size_t f = 0; f < num_filters; f++) {
        float bias = biases[f];

        for (size_t oh = 0; oh < output_height; oh++) {
            for (size_t ow = 0; ow < output_width; ow++) {
                float sum = 0.0f;

                for (size_t kh = 0; kh < kernel_height; kh++) {
                    for (size_t kw = 0; kw < kernel_width; kw++) {
                        for (size_t c = 0; c < depth; c++) {

                            size_t inY = (oh * strides) + kh;
                            size_t inX = (ow * strides) + kw;

                            if (inY < input_height && inX < input_width) {
                                size_t input_idx = ((inY * input_width + inX) * depth + c);
                                size_t kernel_idx = (((f * kernel_height + kh) * kernel_width + kw ) * depth + c);

                                sum += data[input_idx] * kernels[kernel_idx];
                            }
                        }
                    }
                }

                size_t outIndex = ((oh * output_width + ow) * num_filters + f);

                outData[outIndex] = sum + bias;
            }
        }
    }

    return output;
}

Napi::Value ConvolveDelta_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaInput = info[0].As<Napi::Float32Array>();
    IntArray padded_shape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernel_shape = Vectorize(info[2].As<Napi::Array>());
    size_t oH = info[3].As<Napi::Number>().Int32Value();
    size_t oW = info[4].As<Napi::Number>().Int32Value();
    int pointer = info[5].As<Napi::Number>().Int32Value();

    size_t Hp  = padded_shape[0];
    size_t Wp  = padded_shape[1];
    size_t C_in = padded_shape[2];

    size_t F   = kernel_shape[0];
    size_t KH  = kernel_shape[1];
    size_t KW  = kernel_shape[2];
    size_t C_k = kernel_shape[3];

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();

    cl_int err;

    // Upload the padded delta input to the GPU
    cl_mem delta = clCreateBuffer(gpu.context(),CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * Hp * Wp * C_in, deltaInput.Data(), &err);
    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create delta buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Allocate output buffer: shape [oH, oW, C_k]
    size_t outputSize = oH * oW * C_k;
    cl_mem outputBuf = clCreateBuffer(gpu.context(),CL_MEM_WRITE_ONLY,sizeof(float) * outputSize,nullptr,&err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(delta);
        Napi::TypeError::New(env, "Failed to create output buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Rotate kernels and upload to GPU
    auto kernel_array = Rotate_kernels(F, KH, KW, C_k, pointer);
    cl_mem weights = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * F * KH * KW * C_k, kernel_array.data(), &err);

    cl_kernel kernel = gpu.kernel("delta_convolve");

    // Cast size_t dims to int for the kernel (OpenCL int args)
    int iWp   = (int)Wp;
    int iC_in = (int)C_in;
    int iF    = (int)F;
    int iKH   = (int)KH;
    int iKW   = (int)KW;
    int iC_k  = (int)C_k;
    int ioH   = (int)oH;
    int ioW   = (int)oW;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(kernel, 3, sizeof(int),    &iWp);
    clSetKernelArg(kernel, 4, sizeof(int),    &iC_in);
    clSetKernelArg(kernel, 5, sizeof(int),    &iF);
    clSetKernelArg(kernel, 6, sizeof(int),    &iKH);
    clSetKernelArg(kernel, 7, sizeof(int),    &iKW);
    clSetKernelArg(kernel, 8, sizeof(int),    &iC_k);
    clSetKernelArg(kernel, 9, sizeof(int),    &ioH);
    clSetKernelArg(kernel, 10, sizeof(int),   &ioW);

    // One work-item per output element
    size_t global = outputSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // Read result back to host
    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);
    clEnqueueReadBuffer(queue, outputBuf, CL_TRUE, 0, sizeof(float) * outputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(delta);
    clReleaseMemObject(weights);
    clReleaseMemObject(outputBuf);

    return output;
}

Napi::Value ConvolveDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array padded_input_array = info[0].As<Napi::Float32Array>();
    IntArray padded_shape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernel_shape = Vectorize(info[2].As<Napi::Array>());
    size_t oH = info[3].As<Napi::Number>().Int32Value();
    size_t oW = info[4].As<Napi::Number>().Int32Value();
    int pointer = info[5].As<Napi::Number>().Int32Value();

    size_t Hp = padded_shape[0];
    size_t Wp = padded_shape[1];
    size_t C_in = padded_shape[2];

    size_t F = kernel_shape[0];
    size_t KH = kernel_shape[1];
    size_t KW = kernel_shape[2];
    size_t C_k = kernel_shape[3];

    // rotate kernels
    auto rotated_kernels_array = Rotate_kernels(F, KH, KW, C_k, pointer);

    // Raw pointers (faster access)
    float* padded = padded_input_array.Data();
    float* rotatedKernels = rotated_kernels_array.data();

    // Create output array
    Napi::Float32Array output = Napi::Float32Array::New(env, oH * oW * C_k);
    float* out = output.Data();

    // ---- Convolution ----
    #pragma omp for schedule(static)
    for (size_t c_out = 0; c_out < C_k; c_out++) {
        for (size_t h = 0; h < oH; h++) {
            for (size_t w = 0; w < oW; w++) {
                float sum = 0.0f;
                for (size_t kh = 0; kh < KH; kh++) {
                    for (size_t kw = 0; kw < KW; kw++) {
                        for (size_t f = 0; f < F; f++) {
                            size_t ph = h + kh;
                            size_t pw = w + kw;
                            size_t inputIdx  = (ph * Wp + pw) * C_in + f;
                            size_t kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
                            sum += padded[inputIdx] * rotatedKernels[kernelIdx];
                        }
                    }
                }
                out[(h * oW + w) * C_k + c_out] = sum;
            }
        }
    }
    
    return output;
}

/* ==================== Wrappers ======================== */
Napi::Value ConvolveWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return Convolve_GPU(info);
    }

    return Convolve_CPU(info);
}

Napi::Value ConvolveDeltaWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ConvolveDelta_GPU(info);
    }

    return ConvolveDelta_CPU(info);
}

/* ==================== Module exports ======================== */
void ConvolveRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("Convolve", Napi::Function::New(env, ConvolveWrapper));
    exports.Set("ConvolveDelta", Napi::Function::New(env, ConvolveDeltaWrapper));
}
