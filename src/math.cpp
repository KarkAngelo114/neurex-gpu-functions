#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>
#include "global_activations/global_activations.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>

Napi::Value element_wise_mul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>(); // derivative activation output after using layer's own Z
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>(); // the incoming delta
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    std::string layerID = info[3].As<Napi::String>().Utf8Value();
    int arr_length = arr2.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("element_wise_mul");

    cl_mem dActBuffer = gpu.getDAct(modelID, layerID); // cached raw derivative — no upload
    cl_mem output_arr = gpu.getOrCreate_Delta(modelID, layerID, static_cast<size_t>(arr_length)); // final delta, cached for gradient accumulation to consume next
    cl_mem deltaBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_incoming_delta", static_cast<size_t>(arr_length));
    clEnqueueWriteBuffer(queue, deltaBuffer, CL_FALSE, 0, sizeof(float) * arr_length, arr2.Data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dActBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &deltaBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 3, sizeof(int), &arr_length);

    size_t globalSize = (size_t)arr_length;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_element_wise_mul",  arr_length);
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * arr_length, output.Data(), 0, nullptr, nullptr);

    return output;
}

Napi::Value element_wise_mul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    std::string layerID = info[3].As<Napi::String>().Utf8Value();

    int arr_length = arr1.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_element_wise_mul", arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < arr_length; i++) {
        o[i] = a1[i] * a2[i];
    }

    return output;
}

Napi::Value element_wise_sub_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    std::string layerID = info[3].As<Napi::String>().Utf8Value();
    int size = arr1.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_element_wise_sub", size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("element_wise_sub");

    cl_mem input_arr1 = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_element_wise_sub_input_arr1", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input_arr1, CL_FALSE, 0, sizeof(float) * size, arr1.Data(), 0, nullptr, nullptr);

    cl_mem input_arr2 = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_element_wise_sub_input_arr2", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input_arr2, CL_FALSE, 0, sizeof(float) * size, arr2.Data(), 0, nullptr, nullptr);

    cl_mem output_arr = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_element_wise_sub_output_buffer", static_cast<size_t>(size));

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_arr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_arr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    size_t globalSize = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * size, output.Data(), 0, nullptr, nullptr);
    
    return output;
}

Napi::Value element_wise_sub_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    std::string layerID = info[3].As<Napi::String>().Utf8Value();
    int arr_length = arr1.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_element_wise_sub", arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < arr_length; i++) {
        o[i] = a1[i] - a2[i];
    } 

    return output;
}

Napi::Value scaleDiff_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    Napi::Float32Array arr3 = info[2].As<Napi::Float32Array>();
    std::string modelID = info[3].As<Napi::String>().Utf8Value();
    std::string layerID = info[4].As<Napi::String>().Utf8Value();
    int size = arr1.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_scale_diff", size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("scale_diff");

    cl_mem input_arr1 = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_scale_diff_arr1", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input_arr1, CL_FALSE, 0, sizeof(float) * size, arr1.Data(), 0, nullptr, nullptr);

    cl_mem input_arr2 = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_scale_diff_arr2", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input_arr2, CL_FALSE, 0, sizeof(float) * size, arr2.Data(), 0, nullptr, nullptr);

    cl_mem input_arr3 = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_scale_diff_arr3", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input_arr3, CL_FALSE, 0, sizeof(float) * size, arr3.Data(), 0, nullptr, nullptr);

    cl_mem output_arr = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_scale_diff_output_buffer", static_cast<size_t>(size));


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_arr1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_arr2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_arr3);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_arr);
    clSetKernelArg(kernel, 4, sizeof(int), &size);

    size_t globalSize = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // read result
    clEnqueueReadBuffer(queue, output_arr, CL_TRUE, 0, sizeof(float) * size, output.Data(), 0, nullptr, nullptr);
    
    return output;
}

Napi::Value scaleDiff_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    Napi::Float32Array arr3 = info[2].As<Napi::Float32Array>();
    std::string modelID = info[3].As<Napi::String>().Utf8Value();
    std::string layerID = info[4].As<Napi::String>().Utf8Value();
    int arr_length = arr1.ElementLength();

    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_scale_diff", arr_length);
    float scale = 2.0f / arr_length;

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* a3 = arr3.Data();
    float* o = output.Data();

    #pragma omp parallel for
    #pragma omp unroll partial(4)
    for (int i = 0; i < arr_length; i++) {
        o[i] = (a1[i] - a2[i]) * a3[i] * scale;
    }

    return output;
}

Napi::Value Scale_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int size = inputArray.ElementLength();
    int scalingFactor = info[1].As<Napi::Number>().Int32Value();
    std::string modelID = info[2].As<Napi::String>().Utf8Value();
    std::string layerID = info[3].As<Napi::String>().Utf8Value();

    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_scale_output", size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("scale");
    cl_context context = gpu.context();

    cl_mem input = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_scale_input_arr", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, input, CL_FALSE, 0, sizeof(float) * size, inputArray.Data(), 0, nullptr, nullptr);

    // 3. Set Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(int), &scalingFactor);
    clSetKernelArg(kernel, 2, sizeof(int), &size);

    // 4. Execute
    size_t global = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, input, CL_TRUE, 0, sizeof(float) * size, output.Data(), 0, nullptr, nullptr);
    
    return output;
}

Napi::Value Scale_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array inputArray = info[0].As<Napi::Float32Array>();
    int scalingFactor = info[1].As<Napi::Number>().Int32Value();

    float* data = inputArray.Data();
    int length = inputArray.ElementLength();

    int i = 0;
    for (; i + 3 < length; i += 4) {
        data[i] /= scalingFactor;
        data[i + 1] /= scalingFactor;
        data[i + 2] /= scalingFactor;
        data[i + 3] /= scalingFactor;
    }

    for (; i < length; i++) {
        data[i] /= scalingFactor;
    }

    return inputArray;
}

Napi::Value accumulate_element_wise_mul_CPU(const Napi::CallbackInfo& info) {
    Napi::Float32Array inputArray1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array inputArray2 = info[1].As<Napi::Float32Array>();
    Napi::Float32Array inputArray3 = info[2].As<Napi::Float32Array>();
    int size = inputArray1.ElementLength();

    float* arr1 = inputArray1.Data();
    float* arr2 = inputArray2.Data();
    float* arr3 = inputArray3.Data();

    #pragma omp parallel for
    
    for (int i = 0; i < size; i++) {
        arr3[i] += arr2[i] * arr1[i];
    }

    return inputArray3;
}

Napi::Value element_wise_add_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    int size = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, size);
    

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    #pragma omp parallel for
    
    for (int i = 0; i < size; i++) {
        o[i] = a1[i] + a2[i];
    }

    return output;
    
}

Napi::Value element_wise_add_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    std::string globalReferenceID = info[2].As<Napi::String>().Utf8Value();
    std::string localReferenceID = info[2].As<Napi::String>().Utf8Value();

    size_t size = arr1.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, globalReferenceID, localReferenceID, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("element_wise_add");

    cl_mem input1 = gpu.getOrCreate_SomethingToWriteOn(globalReferenceID, localReferenceID+"_element_wise_add_input_1", size);
    cl_mem input2 = gpu.getOrCreate_SomethingToWriteOn(globalReferenceID, localReferenceID+"_element_wise_add_input_2", size);
    cl_mem outputTensor = gpu.getOrCreate_SomethingToWriteOn(globalReferenceID, localReferenceID+"_element_wise_add_output", size);
    clEnqueueWriteBuffer(queue, input1, CL_FALSE, 0, sizeof(float) * size, arr1.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, input2, CL_FALSE, 0, sizeof(float) * size, arr2.Data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputTensor);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    size_t globalSize = (size_t)size;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, outputTensor, CL_TRUE, 0, sizeof(float) * size, output.Data(), 0, nullptr, nullptr);


    return output;
    
}

Napi::Value fusedFinalDeltaProjector_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array zs_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array cachedActivated_output = info[1].As<Napi::Float32Array>();
    Napi::Float32Array incomingDelta = info[2].As<Napi::Float32Array>();
    std::string activationType = info[3].As<Napi::String>().Utf8Value();
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();

    int size = incomingDelta.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID + "_final_delta", size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel activationKernel = gpu.kernel("d"+activationType); // all derivation activation functions has a "d" prefix so that what ever activation type string is passed, it will become like: drelu because "d" + "relu". All activation function kernels has "d" on each function name
    cl_kernel element_wise_kernel = gpu.kernel("element_wise_mul");

    cl_mem z_values = gpu.getZ(modelID, layerID);
    cl_mem cachedActivatedVector = gpu.getActivationOutput(modelID, layerID);
    cl_mem finalDelta = gpu.getOrCreate_Delta(modelID, layerID, static_cast<size_t>(size));
    cl_mem deltaBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_incoming_delta", static_cast<size_t>(size));
    clEnqueueWriteBuffer(queue, deltaBuffer, CL_FALSE, 0, sizeof(float) * arr_length, incomingDelta.Data(), 0, nullptr, nullptr);
    cl_mem dAct = gpu.getOrCreate_DAct(modelID, layerID, static_cast<size_t>(size));

    size_t globalSize = (size_t)size;

    // if the activation function is softmax, then it's derivative is a full Jacobian equation
    if (activationType == "softmax") {

        const float* probabilities = cachedActivated_output.Data();
        const float* delta = incomingDelta.Data();

        float dotProduct = 0.0f;
        #pragma omp unroll partial(4)
        for (int i = 0; i < size; i++) {
            dotProduct += probabilities[i] * delta[i];
        }

        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &cachedActivatedVector);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &deltaBuffer);
        clSetKernelArg(activationKermel, 2, sizeof(cl_mem), &finalDelta);
        clSetKernelArg(activationKernel, 3, sizeof(float), &dotProduct);
        clSetKernelArg(activationKernel, 4, sizeof(int), &size);

        
        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

        clEnqueueReadBuffer(queue, finalDelta, CL_TRUE, 0, sizeof(float)* size, output.Data(), 0, nullptr, nullptr);

        return output;
    }

    // if not a softmax activation function, then proceed to applying the derivative of Z and element-wise multiplication to produce final delta
    clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
    clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &dAct);
    clSetKernelArg(activationKernel, 2, sizeof(int), &size);

    clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    // after running the derivative activation kernel, perform the element_wise_mul with dAct and incoming delta to produce the final delta
    clSetKernelArg(element_wise_kernel, 0, sizeof(cl_mem), &dAct);
    clSetKernelArg(element_wise_kernel, 1, sizeof(cl_mem), &deltaBuffer);
    clSetKernelArg(element_wise_kernel, 2, sizeof(cl_mem), &finalDelta);
    clSetKernelArg(element_wise_kernel, 3, sizeof(int), &size);

    clEnqueueNDRangeKernel(queue, element_wise_kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, finalDelta, CL_TRUE, 0, sizeof(float)* size, output.Data(), 0, nullptr, nullptr);

    return output;

}

Napi::Value fusedFinalDeltaProjector_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array zs_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array cachedActivated_output = info[1].As<Napi::Float32Array>();
    Napi::Float32Array incomingDelta = info[2].As<Napi::Float32Array>();
    std::string activationType = info[3].As<Napi::String>().Utf8Value();
    std::string modelID = info[4].As<Napi::String>().Utf8Value();
    std::string layerID = info[5].As<Napi::String>().Utf8Value();

    int size = incomingDelta.ElementLength();
    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID + "_final_delta", size);

    if (activationType == "softmax") {
        DSOFTMAX(cachedActivated_output, incomingDelta, output);
        return output;
    }

    std::unordered_map<std::string, void(*)(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn)> activations = {
        {"linear", DLINEAR},
        {"relu", DRELU},
        {"sigmoid", DSIGMOID},
        {"tanh", DTANH}
    };

    activations[activationType](zs, output);

    float* derivative = output.Data();
    const float* delta = incomingDelta.Data();

    // element-wise multiplication
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        derivative[i] *= delta[i];
    }

    return output;
}

// ====== wrappers ================

Napi::Value element_wise_mul_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return element_wise_mul_GPU(info);
    }

    return element_wise_mul_CPU(info);
}

Napi::Value element_wise_sub_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return element_wise_sub_GPU(info);
    }

    return element_wise_sub_CPU(info);

}

Napi::Value scaleDiffWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return scaleDiff_GPU(info);
    }

    return scaleDiff_CPU(info);

}

Napi::Value ScalerWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return Scale_GPU(info);
    }

    return Scale_CPU(info);
    
}

Napi::Value accumulate_element_wise_mul_wrapper(const Napi::CallbackInfo& info) {
    return accumulate_element_wise_mul_CPU(info);
}

Napi::Value element_wise_add_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") return element_wise_add_GPU(info);
    return element_wise_add_CPU(info);
}

Napi::Value fusedFinalDeltaProjector_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") return fusedFinalDeltaProjector_GPU(info);
    return fusedFinalDeltaProjector_CPU(info);
}

void Math_OPS(Napi::Env env, Napi::Object exports) {
    exports.Set("element_wise_mul", Napi::Function::New(env, element_wise_mul_wrapper));
    exports.Set("element_wise_sub", Napi::Function::New(env, element_wise_sub_wrapper));
    exports.Set("scaleDiff", Napi::Function::New(env, scaleDiffWrapper));
    exports.Set("scale", Napi::Function::New(env, ScalerWrapper));
    exports.Set("accumulate_element_wise_mul", Napi::Function::New(env, accumulate_element_wise_mul_wrapper));
    exports.Set("element_wise_add", Napi::Function::New(env, element_wise_add_wrapper));
    exports.Set("fusedFinalDeltaProjector", Napi::Function::New(env, fusedFinalDeltaProjector_wrapper));
}