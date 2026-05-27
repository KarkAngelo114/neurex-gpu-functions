#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>
#include <cmath>

Napi::Value ComputeGradientForDenseWeights_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array activation_output = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputSize = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_kernel kernel = gpu.kernel("computeWeightGradsForConnected_Layer");
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem activations = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * activation_output.ElementLength(), activation_output.Data(), nullptr);
    cl_mem deltaInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);
    cl_mem weight_grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &deltaInput);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight_grads);
    clSetKernelArg(kernel, 3, sizeof(int), &inputSize);
    clSetKernelArg(kernel, 4, sizeof(int), &outputSize);

    size_t globalSize[2] = {
        (size_t)inputSize,
        (size_t)outputSize,
    };

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, weight_grads, CL_TRUE, 0, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), 0, nullptr, nullptr);

    clFinish(queue);

    clReleaseMemObject(activations);
    clReleaseMemObject(deltaInput);
    clReleaseMemObject(weight_grads);

    return weightGrads;

}

Napi::Value ComputeGradientForDenseWeights_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array activation_output = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputSize = info[3].As<Napi::Number>().Int32Value();
    int outputSize = info[4].As<Napi::Number>().Int32Value();

    float* a = activation_output.Data();
    float* d = deltas.Data();
    float* wg = weightGrads.Data();

    for (size_t i = 0; i < inputSize; i++) {
        float inputVal = a[i];
        int offset = i * outputSize;
        for (size_t j = 0; j < outputSize; j++) {
            wg[offset + j] += inputVal * d[j];
        }
    }
    return weightGrads;
}

Napi::Value computeBiasGradsForConnected_Layer_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasgrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    int biasGradsSize = biasgrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_kernel kernel = gpu.kernel("computeBiasGradsForConnected_Layer");
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();

    cl_mem gradsInput = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasGradsSize, biasgrads.Data(), nullptr);
    cl_mem deltaInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &deltaInput);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradsInput);
    clSetKernelArg(kernel, 2, sizeof(int), &biasGradsSize);

    size_t globalSize = biasGradsSize;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // Read results back into the output array
    clEnqueueReadBuffer(queue, gradsInput, CL_TRUE, 0, sizeof(float) * biasGradsSize, biasgrads.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(gradsInput);
    clReleaseMemObject(deltaInput);

    return biasgrads;
}

Napi::Value computeBiasGradsForConnected_Layer_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasgrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();

    float* bg = biasgrads.Data();
    float* d = deltas.Data();
    size_t length = biasgrads.ElementLength();

    
    for (size_t i = 0; i < length; i++) {
        bg[i] += d[i];
    }

    return biasgrads;
}

Napi::Value computeKernelGradients_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array delta = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputH = info[3].As<Napi::Number>().Int32Value(); 
    int inputW = info[4].As<Napi::Number>().Int32Value(); 
    int Cin = info[5].As<Napi::Number>().Int32Value(); 
    int H = info[6].As<Napi::Number>().Int32Value(); 
    int W = info[7].As<Napi::Number>().Int32Value(); 
    int Cout = info[8].As<Napi::Number>().Int32Value(); 
    int Kh = info[9].As<Napi::Number>().Int32Value(); 
    int Kw = info[10].As<Napi::Number>().Int32Value();

    int padH = Kh / 2;
    int padW = Kw / 2;

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("computeKernelGradients");

    cl_mem activations = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input.ElementLength(), input.Data(), nullptr);
    cl_mem delta_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * delta.ElementLength(), delta.Data(), nullptr);
    cl_mem gradsArr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gradsArr);
    clSetKernelArg(kernel, 3, sizeof(int), &inputH);
    clSetKernelArg(kernel, 4, sizeof(int), &inputW);
    clSetKernelArg(kernel, 5, sizeof(int), &Cin);
    clSetKernelArg(kernel, 6, sizeof(int), &H);
    clSetKernelArg(kernel, 7, sizeof(int), &W);
    clSetKernelArg(kernel, 8, sizeof(int), &Cout);
    clSetKernelArg(kernel, 9, sizeof(int), &Kh);
    clSetKernelArg(kernel, 10, sizeof(int), &Kw);
    clSetKernelArg(kernel, 11, sizeof(int), &padH);
    clSetKernelArg(kernel, 12, sizeof(int), &padW);

    size_t globalSize[4] = {
        (size_t)Cout,
        (size_t)Kh,
        (size_t)Kw,
        (size_t)Cin
    };

    clEnqueueNDRangeKernel(queue, kernel, 4, nullptr, globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, gradsArr, CL_TRUE, 0, sizeof(float) * weightGrads.ElementLength(), weightGrads.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(activations);
    clReleaseMemObject(delta_input);
    clReleaseMemObject(gradsArr);

    return weightGrads;
}

Napi::Value computeKernelGradients_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array delta = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    int inputH = info[3].As<Napi::Number>().Int32Value(); 
    int inputW = info[4].As<Napi::Number>().Int32Value(); 
    int Cin = info[5].As<Napi::Number>().Int32Value(); 
    int H = info[6].As<Napi::Number>().Int32Value(); 
    int W = info[7].As<Napi::Number>().Int32Value(); 
    int Cout = info[8].As<Napi::Number>().Int32Value(); 
    int Kh = info[9].As<Napi::Number>().Int32Value(); 
    int Kw = info[10].As<Napi::Number>().Int32Value();

    float* input_data = input.Data();
    float* d = delta.Data();
    float* wg = weightGrads.Data();

    int padH = Kh / 2;
    int padW = Kw / 2;

    for (size_t f = 0; f < Cout; f++) {
        for (size_t kh = 0; kh < Kh; kh++) {
            for (size_t kw = 0; kw < Kw; kw++) {
                for (size_t c = 0; c < Cin; c++) {
                    float sum = 0.0f;
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            int inH = h + kh - padH;
                            int inW = w + kw - padW;

                            if (inH >= 0 && inH < inputH && inW >= 0 && inW < inputW) {

                                size_t inputIndex = (inH * inputW + inW) * Cin + c;

                                size_t deltaIndex = (h * W + w) * Cout + f;

                                sum += input_data[inputIndex] * d[deltaIndex];
                            }
                        }
                    }
                    size_t gradIndex = ((f * Kh + kh) * Kw + kw) * Cin + c;

                    wg[gradIndex] += sum;
                }
            }
        }
    }
    return weightGrads;
}

Napi::Value computeBiasGradsForConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    int outH = info[2].As<Napi::Number>().Int32Value();
    int outW = info[3].As<Napi::Number>().Int32Value();
    int numFilters = info[4].As<Napi::Number>().Int32Value();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("computeBiasGradsForConv");
    
    cl_mem grads = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * biasGrads.ElementLength(), biasGrads.Data(), nullptr);
    cl_mem delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * deltas.ElementLength(), deltas.Data(), nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &grads);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 2, sizeof(int), &outH);
    clSetKernelArg(kernel, 3, sizeof(int), &outW);
    clSetKernelArg(kernel, 4, sizeof(int), &numFilters);

    size_t globalSize = (size_t)numFilters;

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(queue, grads, CL_TRUE, 0, sizeof(float) * biasGrads.ElementLength(), biasGrads.Data(), 0, nullptr, nullptr);
    clFinish(queue);
    clReleaseMemObject(grads);
    clReleaseMemObject(delta);

    return biasGrads;

}

Napi::Value computeBiasGradsForConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    size_t outH = info[2].As<Napi::Number>().Int32Value();
    size_t outW = info[3].As<Napi::Number>().Int32Value();
    int numFilters = info[4].As<Napi::Number>().Int32Value();

    float* bg = biasGrads.Data();
    float* d = deltas.Data();

    for (size_t f = 0; f < numFilters; f++) {
        float sum = 0.0f;

        for (size_t h = 0; h < outH; h++) {
            for (size_t w = 0; w < outW; w++) {
                size_t idx = (h * outW + w) * numFilters + f;
                sum += d[idx];
            }
        }
        bg[f] += sum;
    }

    return biasGrads;
}

Napi::Value ComputeTransKernelGrads_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();


}

Napi::Value ComputeTransKernelGrads_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array dilatedInput_arr = info[0].As<Napi::Float32Array>();
    Napi::Float32Array delta_arr = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads_array = info[2].As<Napi::Float32Array>();
    int dilatedH = info[3].As<Napi::Number>().Int32Value();
    int dilatedW = info[4].As<Napi::Number>().Int32Value(); 
    int inputDepth = info[5].As<Napi::Number>().Int32Value(); 
    int OutputHeight = info[6].As<Napi::Number>().Int32Value(); 
    int OutputWidth = info[7].As<Napi::Number>().Int32Value();
    int OutputDepth = info[8].As<Napi::Number>().Int32Value(); 
    int filters = info[9].As<Napi::Number>().Int32Value(); 
    int kernelHeight = info[10].As<Napi::Number>().Int32Value(); 
    int kernelWidth = info[11].As<Napi::Number>().Int32Value();
    int padH = kernelHeight / 2;
    int padW = kernelWidth / 2;


    float* dilatedInput = dilatedInput_arr.Data();
    float* delta = delta_arr.Data();
    float* weightGrads = weightGrads_array.Data();

    for (int f = 0; f < filters; f++) {
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                for (int c = 0; c < inputDepth; c++) {
                    
                    float sum = 0.0f;

                    for (int h = 0; h < OutputHeight; h++) {
                        for (int w = 0; w < OutputWidth; w++) {

                            int inH = h + kh - padH;
                            int inW = w + kw - padW;

                            if (inH >= 0 && inH < dilatedH && inW >= 0 && inW < dilatedW) {
                                int inputIndex = (inH * dilatedW + inW) * inputDepth + c;
                                int deltaIndex = (h * OutputWidth + w) * OutputDepth + f;

                                sum += dilatedInput[inputIndex] * delta[deltaIndex];
                            }
                        }
                    }

                    int gradIndex = ((f * kernelHeight + kh) * kernelWidth + kw) * inputDepth + c;
                    weightGrads[gradIndex] += sum;
                }
            }
        }
    }


    return weightGrads;

}

Napi::Value computeBiasGradsForConnected_LayerWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeBiasGradsForConnected_Layer_GPU(info);
    }

    return computeBiasGradsForConnected_Layer_CPU(info);
}

Napi::Value ComputeGradientForDenseWeightsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ComputeGradientForDenseWeights_GPU(info);
    }
    return ComputeGradientForDenseWeights_CPU(info);
}

Napi::Value computeKernelGradientsWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeKernelGradients_GPU(info);
    }

    return computeKernelGradients_CPU(info);
}

Napi::Value computeBiasGradsForConvWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return computeBiasGradsForConv_GPU(info);
    }

    return computeBiasGradsForConv_CPU(info);
}

Napi::Value computeTransKernelGradients_wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return ComputeTransKernelGrads_GPU(info);
    }
    return ComputeTransKernelGrads_GPU(info);
}

/* ================ module exports ===================*/
void GradientCalculationRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("computeWeightGradientsForWeightsInConnectedLayer", Napi::Function::New(env, ComputeGradientForDenseWeightsWrapper));
    exports.Set("computeKernelGradients", Napi::Function::New(env, computeKernelGradientsWrapper));
    exports.Set("computeBiasGradsForConnected_Layer", Napi::Function::New(env, computeBiasGradsForConnected_LayerWrapper));
    exports.Set("computeBiasGradsForConv", Napi::Function::New(env, computeBiasGradsForConvWrapper));
    exports.Set("computeTransKernelGradients", Napi::Function::New(env, computeTransKernelGradients_wrapper));
}