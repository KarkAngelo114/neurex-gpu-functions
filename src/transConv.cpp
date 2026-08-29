#include <napi.h>
#include <CL/cl.h>
#include "globals/globals.h"
#include "gpu/gpu_context.h"
#include <vector>
#include <cmath>

using IntArray = std::vector<int>;


static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

Napi::Value transConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[7].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    size_t outputSize = (size_t)oH * (size_t)oW * (size_t)f;
    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, outputSize);

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("transConv");

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputTensor.ElementLength(), inputTensor.Data(), nullptr);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightsArray.ElementLength(), weightsArray.Data(), nullptr);
    cl_mem biases = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasesArray.ElementLength(), biasesArray.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputSize, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 4, sizeof(int), &iH);
    clSetKernelArg(kernel, 5, sizeof(int), &iW);
    clSetKernelArg(kernel, 6, sizeof(int), &iD);
    clSetKernelArg(kernel, 7, sizeof(int), &oH);
    clSetKernelArg(kernel, 8, sizeof(int), &oW);
    clSetKernelArg(kernel, 9, sizeof(int), &f);
    clSetKernelArg(kernel, 10, sizeof(int), &kh);
    clSetKernelArg(kernel, 11, sizeof(int), &kw);
    clSetKernelArg(kernel, 12, sizeof(int), &d);
    clSetKernelArg(kernel, 13, sizeof(int), &strides);
    clSetKernelArg(kernel, 14, sizeof(int), &padTop);
    clSetKernelArg(kernel, 15, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        (size_t)oH,
        (size_t)oW,
        (size_t)f
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * outputSize, outputTensor.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input);
    clReleaseMemObject(weights);
    clReleaseMemObject(biases);
    clReleaseMemObject(output);

    return outputTensor;
}

Napi::Value transConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();
    Napi::Float32Array biasesArray = info[7].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, oH * oW * f);

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    // helper functions
    auto inputIndex = [&](int y, int x, int c) {
        return (y * iW + x) * iD + c;
    };
    auto outputIndex = [&](int y, int x, int c) {
        return (y * oW + x) * f + c;
    };
    auto weightIndex = [&](int filter, int ky, int kx, int c) {
        return (((filter * kh) + ky) * kw + kx) * d + c;
    };

    float* input = inputTensor.Data();
    float* weights = weightsArray.Data();
    float* biases = biasesArray.Data(); 
    float* output = outputTensor.Data();
    

    // set biases before hand
    for (int y = 0; y < oH; y++) {
        for (int x = 0; x < oW; x++) {

            int outputBase = (y * oW + x) * f;

            for (int filter = 0; filter < f; filter++) {
                output[outputBase + filter] = biases[filter]; // for safety, we use "=" rather "+="
            }
        }
    }

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {

            int inputBase = (iy * iW + ix) * iD;

            for (int ky = 0; ky < kh; ky++) {

                int oy = iy * strides + ky - padTop;

                // Kernel row falls outside output.
                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {

                    int ox = ix * strides + kx - padLeft;

                    // Kernel column falls outside output.
                    if (ox < 0 || ox >= oW) continue;

                    int outputBase = (oy * oW + ox) * f;

                    /*
                     * For every output filter, accumulate the
                     * input channels multiplied by the kernel.
                     */
                    for (int filter = 0; filter < f; filter++) {

                        float sum = 0.0f;

                        int weightBase = ((filter * kh + ky) * kw + kx) * d;

                        for (int c = 0; c < d; c++) {
                            sum += input[inputBase + c] * weights[weightBase + c];
                        }

                        output[outputBase + filter] += sum; // we use "+=" to sum to the bias
                    }
                }
            }
        }
    }

    return outputTensor;
}

Napi::Value transConvBackward_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, iH * iW * iD);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("transConvBackward");

    cl_mem delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltaTensor.ElementLength(), deltaTensor.Data(), nullptr);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightsArray.ElementLength(), weightsArray.Data(), nullptr);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputTensor.ElementLength(), nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 3, sizeof(int), &iH);
    clSetKernelArg(kernel, 4, sizeof(int), &iW);
    clSetKernelArg(kernel, 5, sizeof(int), &iD);
    clSetKernelArg(kernel, 6, sizeof(int), &oH);
    clSetKernelArg(kernel, 7, sizeof(int), &oW);
    clSetKernelArg(kernel, 8, sizeof(int), &oD);
    clSetKernelArg(kernel, 9, sizeof(int), &f);
    clSetKernelArg(kernel, 10, sizeof(int), &kh);
    clSetKernelArg(kernel, 11, sizeof(int), &kw);
    clSetKernelArg(kernel, 12, sizeof(int), &d);
    clSetKernelArg(kernel, 13, sizeof(int), &strides);
    clSetKernelArg(kernel, 14, sizeof(int), &padTop);
    clSetKernelArg(kernel, 15, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        (size_t)iH,
        (size_t)iW,
        (size_t)iD
    };

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * outputTensor.ElementLength(), outputTensor.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(delta);
    clReleaseMemObject(weights);
    clReleaseMemObject(output);

    return outputTensor;
}

Napi::Value transConvBackward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array deltaTensor = info[0].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[1].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[2].As<Napi::Array>());
    int strides = info[3].As<Napi::Number>().Int32Value();
    int filters = info[4].As<Napi::Number>().Int32Value();
    IntArray weightShape = Vectorize(info[5].As<Napi::Array>());
    Napi::Float32Array weightsArray = info[6].As<Napi::Float32Array>();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int f = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = std::floor(padH / 2);
    int padLeft = std::floor(padW / 2);

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, iH * iW * iD);

    // helper function
    auto deltaOutputIndex = [&](int y, int x, int f) {
        return (y * oW + x) * oD + f;
    };

    float* delta = deltaTensor.Data();
    float* weights = weightsArray.Data();
    float* deltaInput = outputTensor.Data();

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {
            for (int ky = 0; ky < kh; ky++) {
                int oy = iy * strides + ky - padTop;

                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {

                    int ox = ix * strides + kx - padLeft;

                    if (ox < 0 || ox >= oW) continue;

                    for (int filter = 0; filter < f; filter++) {

                        float deltaY = delta[deltaOutputIndex(oy, ox, filter)];
                        int weightBase = ((filter * kh + ky) * kw + kx) * d;
                        int inputBase = (iy * iW + ix) * iD;

                        for (int c = 0; c < d; c++) {

                            deltaInput[inputBase + c] += deltaY * weights[weightBase + c];
                        }
                    }
                }
            }
        }
    }

    return outputTensor;
}

Napi::Value transConvWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return transConv_GPU(info);
    }
    return transConv_CPU(info);
}

Napi::Value transConvBackwardWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return transConvBackward_GPU(info);
    }
    return transConvBackward_CPU(info);
}

void transConvFunc(Napi::Env env, Napi::Object exports) {
    exports.Set("transConv", Napi::Function::New(env, transConvWrapper));
    exports.Set("transConvBackward", Napi::Function::New(env, transConvBackwardWrapper));
}