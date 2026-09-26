#include <napi.h>
#include <omp.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <CL/cl.h>
#include <cmath>


Napi::Value GradientClipping_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputGrads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    int size = inputGrads.ElementLength();

    float* grads = inputGrads.Data();

    float norm = 0.0f;

    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += grads[i] * grads[i];
    }

    norm = std::sqrt(norm);

    if (norm > threshold) {
        float scalingVal = threshold / norm;

        auto& gpu = GpuContext::instance();
        cl_command_queue queue = gpu.queue();
        cl_context context = gpu.context();
        cl_kernel kernel = gpu.kernel("gradientClipping");

        cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, inputGrads.Data(), nullptr);
        
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
        clSetKernelArg(kernel, 1, sizeof(float), &scalingVal);
        clSetKernelArg(kernel, 2, sizeof(int), &size);

        size_t globalSize = (size_t)size;
        clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, input, CL_TRUE, 0, sizeof(int)* size, inputGrads.Data(), 0, nullptr, nullptr);
        clReleaseMemObject(input);

        return inputGrads;
    }

    return inputGrads;

}

Napi::Value GradientClipping_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputGrads = info[0].As<Napi::Float32Array>();
    float threshold = info[1].As<Napi::Number>().FloatValue();
    int size = inputGrads.ElementLength();

    float* grads = inputGrads.Data();

    float norm = 0.0f;

    #pragma omp simd reduction(+:norm)
    for (int i = 0; i < size; i++) {
        norm += grads[i] * grads[i];
    }

    norm = std::sqrt(norm);

    if (norm > threshold) {
        float scalingVal = threshold / norm;

        #pragma omp unroll partial(4)
        for (int i = 0; i < size; i++) {
            grads[i] *= scalingVal;
        }
    }

    return inputGrads;
}

Napi::Value LayerNorm_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int size = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array gammaTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array betaTensor = info[3].As<Napi::Float32Array>();
    float eps = info[4].As<Napi::Number>().FloatValue();
    int pointer = info[5].As<Napi::Number>().Int32Value();
    std::string modelID = info[6].As<Napi::String>().Utf8Value();

    float* input = inputTensor.Data();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("layer_norm_standard_size");

    // compute mean
    float mean = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }

    // scale mean
    mean /= size;

    // compute variance
    float variance = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }

    // scale variance
    variance /= size;
    
    // get standardization value by getting the square root of sum of variance and epsilon
    float std = std::sqrt(variance + eps);

    cl_mem _input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* size, inputTensor.Data(), nullptr);
    cl_mem _gamma = gpu.getWeights(modelID, pointer);
    cl_mem _beta = gpu.getBiases(modelID, pointer);
    cl_mem output = gpu.getOrCreate_ActivationOutput(modelID, pointer, static_cast<size_t>(size)); // since there's no activation function in a layer norm, we cached the final output and treat it as an activation_output to be used by the gradient accumulation function

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_gamma);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_beta);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 4, sizeof(float), &mean);
    clSetKernelArg(kernel, 5, sizeof(float), &std);
    clSetKernelArg(kernel, 6, sizeof(int), &size);

    size_t globalSize = (size_t)size;

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, size);
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float)* size, outputTensor.Data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(_input);

    return outputTensor;
}

Napi::Value LayerNorm_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    int size = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array gammaTensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array betaTensor = info[3].As<Napi::Float32Array>();
    float eps = info[4].As<Napi::Number>().FloatValue();

    Napi::Float32Array outputTensor = Napi::Float32Array::New(env, size);

    float* input = inputTensor.Data();
    float* gamma = gammaTensor.Data();
    float* beta = betaTensor.Data();
    float* output = outputTensor.Data();

    float mean = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }

    mean /= size;

    float variance = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }

    variance /= size;

    float std = std::sqrt(variance + eps);

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        float xHat = (input[i] - mean) / std;
        output[i] = gamma[i] * xHat + beta[i];
    }

    return outputTensor;
}

Napi::Value LayerNormBackward_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array dy = info[0].As<Napi::Float32Array>();
    Napi::Float32Array x = info[1].As<Napi::Float32Array>();
    Napi::Float32Array gamma = info[2].As<Napi::Float32Array>(); // this will not be used here
    int size = info[3].As<Napi::Number>().Int32Value();
    int pointer = info[4].As<Napi::Number>().Int32Value();
    std::string modelID = info[5].As<Napi::String>().Utf8Value();

    if (size <= 0 ||dy.ElementLength() < static_cast<size_t>(size) || x.ElementLength() < static_cast<size_t>(size)) {
        Napi::TypeError::New(env,"Invalid layer normalization input size").ThrowAsJavaScriptException();
        return env.Null();
    }

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("layer_norm_backward_one");


    cl_mem xBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, x.Data(), nullptr);
    cl_mem dyBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, dy.Data(), nullptr);
    cl_mem gammaBuffer = gpu.getWeights(modelID, pointer);
    cl_mem dxBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, nullptr);
    cl_mem dgammaBuffer = gpu.getOrCreate_dGamma(modelID, pointer, static_cast<size_t>(size)); // create cacheable dGamma
    cl_mem dbetaBuffer = gpu.getOrCreate_dBeta(modelID, pointer, static_cast<size_t>(size)); // create cacheable dBeta

    const float eps = 1.0e-5f;

    // Find the device and a legal power-of-two workgroup size.
    cl_device_id device = nullptr;

    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr);

    size_t deviceMaxWorkGroup = 1;
    size_t kernelMaxWorkGroup = 1;

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(deviceMaxWorkGroup), &deviceMaxWorkGroup, nullptr);

    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMaxWorkGroup), &kernelMaxWorkGroup, nullptr);

    size_t maxWorkGroup = deviceMaxWorkGroup;

    if (kernelMaxWorkGroup < maxWorkGroup) {
        maxWorkGroup = kernelMaxWorkGroup;
    }

    size_t localSize = 1;
    size_t limit = static_cast<size_t>(size);

    if (limit > maxWorkGroup) {
        limit = maxWorkGroup;
    }

    while ((localSize * 2) <= limit) {
        localSize *= 2;
    }

    // Arguments 0-7 are normal arguments. Argument 8 is dynamic local memory.
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &xBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dyBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gammaBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &dxBuffer);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &dgammaBuffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &dbetaBuffer);
    clSetKernelArg(kernel, 6, sizeof(int), &size);
    clSetKernelArg(kernel, 7, sizeof(float), &eps);

    // One float per work item in the workgroup.
    clSetKernelArg(kernel,8, localSize * sizeof(float), nullptr);
    size_t globalSize = localSize;

    cl_int err = CL_SUCCESS;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);

    if (err != CL_SUCCESS) {
        clReleaseMemObject(xBuffer);
        clReleaseMemObject(dyBuffer);
        clReleaseMemObject(dxBuffer);
        clReleaseMemObject(dgammaBuffer);
        clReleaseMemObject(dbetaBuffer);

        Napi::Error::New(env, "Failed to enqueue layer norm backward kernel").ThrowAsJavaScriptException();

        return env.Null();
    }

    Napi::Float32Array dx = Napi::Float32Array::New(env, size);
    Napi::Float32Array dgamma = Napi::Float32Array::New(env, size);
    Napi::Float32Array dbeta = Napi::Float32Array::New(env, size);

    // Blocking reads also wait for the kernel to finish.
    clEnqueueReadBuffer(queue, dxBuffer, CL_TRUE, 0, sizeof(float) * size, dx.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, dgammaBuffer, CL_TRUE, 0, sizeof(float) * size, dgamma.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, dbetaBuffer, CL_TRUE, 0, sizeof(float) * size, dbeta.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(xBuffer);
    clReleaseMemObject(dyBuffer);
    clReleaseMemObject(dxBuffer);

    Napi::Object output = Napi::Object::New(env);
    output.Set("dX", dx);
    output.Set("dGamma", dgamma);
    output.Set("dBeta", dbeta);

    return output;
}

Napi::Value LayerNormBackward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dy = info[0].As<Napi::Float32Array>();
    Napi::Float32Array x = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Gamma = info[2].As<Napi::Float32Array>();
    int size = info[3].As<Napi::Number>().Int32Value();
    int pointer = info[4].As<Napi::Number>().Int32Value();
    std::string modelID = info[5].As<Napi::String>().Utf8Value();

    Napi::Float32Array dx = Napi::Float32Array::New(env, size);
    Napi::Float32Array dgamma = Napi::Float32Array::New(env, size);
    Napi::Float32Array dbeta = Napi::Float32Array::New(env, size);

    float* dY = dy.Data();
    float* input = x.Data();
    float* gamma = Gamma.Data();
    float* dGamma = dgamma.Data();
    float* dBeta = dbeta.Data();
    float* dX = dx.Data();

    // 1. Recompute forward statistics (Mean & Variance)
    float mean = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }

    mean /= size;

    float variance = 0.0f;
    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        const float difference = input[i] - mean;
        variance += difference * difference;
    }

    variance /= size;

    const float eps = 1.0e-5f;
    const float stdInv = 1.0f / std::sqrt(variance + eps);

    // 2. Compute normalized values (xHat) and intermediate parameter gradients
    Napi::Float32Array xHat = Napi::Float32Array::New(env, size);
    float* normalized = xHat.Data();
    float sumDy = 0.0f;
    float sumDyXhat = 0.0f;

    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        normalized[i] = (input[i] - mean) * stdInv;
        
        // Parameter Gradients
        dBeta[i] = dY[i];
        dGamma[i] = dY[i] * normalized[i];

        // Accumulate scalar sums for input gradient equation
        const float dyGamma = dY[i] * gamma[i];
        sumDy += dyGamma;
        sumDyXhat += dyGamma * normalized[i];
    }

    // 3. Compute Input Gradient (dX) using closed-form formula
    const float invSize = 1.0f / static_cast<float>(size);

    #pragma omp unroll partial(4)
    for (int i = 0; i < size; i++) {
        const float dyGamma = dY[i] * gamma[i];
        dX[i] = stdInv * (dyGamma - sumDy * invSize - normalized[i] * sumDyXhat * invSize);
    }


    Napi::Object output = Napi::Object::New(env);
    output.Set("dX", dx);
    output.Set("dGamma", dgamma);
    output.Set("dBeta", dbeta);

    return output;
}

Napi::Value gradientClippingWrapper(const Napi::CallbackInfo& info) {
    return GradientClipping_CPU(info);
}

Napi::Value LayerNormWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return LayerNorm_GPU(info);
    }

    return LayerNorm_CPU(info);
}

Napi::Value LayerNorBackwardmWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return LayerNormBackward_GPU(info);
    }
    return LayerNormBackward_CPU(info);
}

void normalizers(Napi::Env env, Napi::Object exports) {
    exports.Set("gradientClipping", Napi::Function::New(env, gradientClippingWrapper));
    exports.Set("computelayerNorm", Napi::Function::New(env, LayerNormWrapper));
    exports.Set("computeLayerNormBackward", Napi::Function::New(env, LayerNorBackwardmWrapper));
}