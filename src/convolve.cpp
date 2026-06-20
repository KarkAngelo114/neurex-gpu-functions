#include <napi.h>
#include <omp.h>
#include <CL/cl.h>
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

static FloatArray Rotate_kernels(int F, int KH, int KW, int D, int pointer) {
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
    cl_command_queue queue = gpu.queue();

    cl_mem inputTensor = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_height * input_width * depth, input.Data(), &err);

    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Failed to create input buffer").ThrowAsJavaScriptException();
        return env.Null();
    }

    cl_mem weights = gpu.weight(pointer);
    cl_mem biases = gpu.bias(pointer);
    cl_mem output_tensor = gpu.output(outputPointer);

    cl_kernel kernel = gpu.kernel("convolve");

    // Set args
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

    size_t global[3] = {
        (size_t)output_height,
        (size_t)output_width,
        (size_t)num_filters
    };

    err = clEnqueueNDRangeKernel(queue,kernel,3,nullptr,global,nullptr,0,nullptr,nullptr);

    if (err != CL_SUCCESS) {
        Napi::TypeError::New(env, "Kernel launch failed").ThrowAsJavaScriptException();
        return env.Null();
    }

    clFinish(queue); // ensure completion

    int expected_size = output_height * output_width * num_filters;
    Napi::Float32Array output = Napi::Float32Array::New(env, expected_size);

    clEnqueueReadBuffer( queue, output_tensor,CL_TRUE, 0, sizeof(float) * expected_size, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputTensor);

    return output;
}

Napi::Value Convolve_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    int pointer = info[5].As<Napi::Number>().Int32Value();

    int numFilters = kernelShape[0];
    int kernelH = kernelShape[1];
    int kernelW = kernelShape[2];
    int depth = kernelShape[3];

    int inputH = inputShape[0];
    int inputW = inputShape[1];
    int outputH = outputShape[0];
    int outputW = outputShape[1];
    int outputSize = outputH * outputW * numFilters;
    int kernelSize = kernelH * kernelW * depth;
    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);
    FloatArray weightsArray = getGlobalWeights(pointer);
    FloatArray biasesArray = getGlobalBiases(pointer);

    float* input = inputTensor.Data();
    float* output = outputTensor.Data();
    float* weights = weightsArray.data();
    float* biases = biasesArray.data();

    for (int y = 0; y < outputH; y++) {
        const baseY = y * strides;

        for (int x = 0; x < outputW; x++) {
            const baseX = x * strides;
            const outBase = (y * outputW + x) * numFilters;

            for (int f = 0; f < numFilters; f++) {
                float sum = biases[f];
                int filterOffset = f * kernelSize;

                for (int ky = 0; ky < kernelH; ky++) {
                    int inY = baseY + ky;

                    if (inY >= inputH) continue;

                    for (int kx = 0; kx < kernelW; kx++) {
                        const inX = baseX + kx;
                        if (inX >= inputW) continue;

                        int inputBase = (inY * inputW + inX) * depth;
                        int kernelBase = filterOffset + (ky * kernelW + kx) * depth;
                        int c = 0;

                        for (; c <= depth - 4; c += 4) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                            sum += input[inputBase + c + 1] * weights[kernelBase + c + 1];
                            sum += input[inputBase + c + 2] * weights[kernelBase + c + 2];
                            sum += input[inputBase + c + 3] * weights[kernelBase + c + 3];
                        }

                        for (; c < depth; c++) {
                            sum += input[inputBase + c] * weights[kernelBase + c];
                        }
                    }
                }
                output[outBase + f] = sum;
            }
        }
    }
    return outputTensor;
}

Napi::Value ConvolveDelta_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaInput = info[0].As<Napi::Float32Array>();
    IntArray padded_shape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernel_shape = Vectorize(info[2].As<Napi::Array>());
    size_t oH = info[3].As<Napi::Number>().Int32Value();
    size_t oW = info[4].As<Napi::Number>().Int32Value();
    int pointer = info[5].As<Napi::Number>().Int32Value();
    int stride = info[6].As<Napi::Number>().Int32Value();

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
    clSetKernelArg(kernel, 11, sizeof(int), &stride);


    // size_t global[3] = outputSize;
    size_t global[3] = {
        (size_t)oH,
        (size_t)oW,
        (size_t)C_k
    };
    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);

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
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray deltaShape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[2].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[3].As<Napi::Array>());
    int pointer = info[4].As<Napi::Number>().Int32Value();
    int stride = info[5].As<Napi::Number>().Int32Value();

    int Hp = deltaShape[0];
    int Wp = deltaShape[1];
    int C_in = deltaShape[2];

    int F = kernelShape[0];
    int KH = kernelShape[1];
    int KW = kernelShape[2];
    int C_k = kernelShape[3];

    int oH = outputShape[0];
    int oW = outputShape[1];

    FloatArray kernels = Rotate_kernels(F, KH, KW, C_k, pointer);

    int H = Hp - KH + 1;
    int W = Wp - KW + 1;
    
    int outputSize = oH * oW * C_k;
    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);

    float* input = inputTensor.Data();
    float* rotated_kernel = kernels.data();
    float* output = outputTensor.Data();

    for (int c_out = 0; c_out < C_k; c_out++) {
        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                float sum = 0.0f;
                for (int kh = 0; kh < KH; kh++) {
                    for (let kw = 0; kw < KW; kw++) {
                        int ph = h * stride + kh;
                        int pw = w * stride + kw;
                        int baseIdx = (ph * Wp + pw) * C_in;
                        int kernelBase = ((kh * KW + kw) * F) * C_k + c_out;

                        int f = 0;
                        for (; f <= F - 4; f += 4) {
                            sum += input[baseIdx + f] * rotated_kernel[f * C_k + kernelBase];
                            sum += input[baseIdx + f + 1] * rotated_kernel[(f + 1) * C_k + kernelBase];
                            sum += input[baseIdx + f + 2] * rotated_kernel[(f + 2) * C_k + kernelBase];
                            sum += input[baseIdx + f + 3] * rotated_kernel[(f + 3) * C_k + kernelBase];
                        }

                        for (; f < F; f++) {
                            int padIdx = baseIdx + f;
                            int kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
                            sum += input[padIdx] * rotated_kernel[kernelIdx];
                        }
                    }
                }
                output[(h * oW + w) * C_k + c_out] = sum;
            }
        }
    }
    return outputTensor;
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
