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

static Napi::Float32Array Rotate_kernels(Napi::Env env, int F, int KH, int KW, int D, const Napi::Float32Array& kernels) {
    size_t kernel_length = kernels.ElementLength();

    Napi::Float32Array output = Napi::Float32Array::New(env, kernel_length);

    const float* src = kernels.Data();
    float* dst = output.Data();

    for (size_t f = 0; f < (size_t)F; f++) {
        for (size_t kh = 0; kh < (size_t)KH; kh++) {
            for (size_t kw = 0; kw < (size_t)KW; kw++) {
                for (size_t d = 0; d < (size_t)D; d++) {
                    // Original Index
                    size_t oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    
                    // Rotated Index (Flip KH and KW)
                    size_t newKh = KH - 1 - kh;
                    size_t newKw = KW - 1 - kw;
                    size_t newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    dst[newIdx] = src[oldIdx];
                }
            }
        }
    }

    return output;
}


// ==================== MAIN FUNCTIONS ======================= //
Napi::Value Convolve_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[5].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[6].As<Napi::Float32Array>();
    int outputPointer = info[7].As<Napi::Number>().Int32Value();

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

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();

    cl_mem inputTensor = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputH * inputW * depth, input.Data(), nullptr);
    cl_mem weights = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightsArray.ElementLength(), weightsArray.Data(), nullptr);
    cl_mem biases = clCreateBuffer(gpu.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasesArray.ElementLength(), biasesArray.Data(), nullptr);
    cl_mem output_tensor = gpu.output(outputPointer);

    cl_kernel kernel = gpu.kernel("convolve");

    // Set args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_tensor);
    clSetKernelArg(kernel, 4, sizeof(int), &strides);
    clSetKernelArg(kernel, 5, sizeof(int), &outputH);
    clSetKernelArg(kernel, 6, sizeof(int), &outputW);
    clSetKernelArg(kernel, 7, sizeof(int), &numFilters);
    clSetKernelArg(kernel, 8, sizeof(int), &kernelH);
    clSetKernelArg(kernel, 9, sizeof(int), &kernelW);
    clSetKernelArg(kernel, 10, sizeof(int), &depth);
    clSetKernelArg(kernel, 11, sizeof(int), &inputH);
    clSetKernelArg(kernel, 12, sizeof(int), &inputW);

    size_t global[3] = {
        (size_t)outputH,
        (size_t)outputW,
        (size_t)numFilters
    };

    clEnqueueNDRangeKernel(queue,kernel,3,nullptr,global,nullptr,0,nullptr,nullptr);

    Napi::Float32Array output = Napi::Float32Array::New(env, outputSize);

    clEnqueueReadBuffer( queue, output_tensor, CL_TRUE, 0, sizeof(float) * outputSize, output.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(inputTensor);
    clReleaseMemObject(weights);
    clReleaseMemObject(biases);

    return output;
}

Napi::Value Convolve_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[3].As<Napi::Array>());
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[5].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[6].As<Napi::Float32Array>();


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

    float* input = inputTensor.Data();
    float* output = outputTensor.Data();
    float* weights = weightsArray.Data();
    float* biases = biasesArray.Data();

    for (int y = 0; y < outputH; y++) {
        int baseY = y * strides;

        for (int x = 0; x < outputW; x++) {
            int baseX = x * strides;
            int outBase = (y * outputW + x) * numFilters;

            for (int f = 0; f < numFilters; f++) {
                float sum = biases[f];
                int filterOffset = f * kernelSize;

                for (int ky = 0; ky < kernelH; ky++) {
                    int inY = baseY + ky;

                    if (inY >= inputH) continue;

                    for (int kx = 0; kx < kernelW; kx++) {
                        int inX = baseX + kx;
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
    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray deltaShape = Vectorize(info[1].As<Napi::Array>());
    IntArray kernelShape = Vectorize(info[2].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[3].As<Napi::Array>());
    Napi::Float32Array kernelsArray = info[4].As<Napi::Float32Array>();
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

    Napi::Float32Array kernels = Rotate_kernels(env, F, KH, KW, C_k, kernelsArray);

    int outputSize = oH * oW * C_k;
    
    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("delta_convolve");

    cl_int err = CL_SUCCESS;
    cl_mem delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * Hp * Wp * C_in, inputTensor.Data(), &err);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * kernels.ElementLength(), kernels.Data(), &err);
    cl_mem outputBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputSize, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &Wp);
    clSetKernelArg(kernel, 4, sizeof(int), &C_in);
    clSetKernelArg(kernel, 5, sizeof(int), &F);
    clSetKernelArg(kernel, 6, sizeof(int), &KH);
    clSetKernelArg(kernel, 7, sizeof(int), &KW);
    clSetKernelArg(kernel, 8, sizeof(int), &C_k);
    clSetKernelArg(kernel, 9, sizeof(int), &oH);
    clSetKernelArg(kernel, 10, sizeof(int), &oW);
    clSetKernelArg(kernel, 11, sizeof(int), &stride);

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
    Napi::Float32Array kernelArray = info[4].As<Napi::Float32Array>();
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

    Napi::Float32Array kernels = Rotate_kernels(env, F, KH, KW, C_k, kernelArray);

    
    int outputSize = oH * oW * C_k;
    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);

    float* input = inputTensor.Data();
    float* rotated_kernel = kernels.Data();
    float* output = outputTensor.Data();

    for (int c_out = 0; c_out < C_k; c_out++) {
        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                float sum = 0.0f;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
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
