
#include <napi.h>
#include <omp.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "global_activations/global_activations.h"
#include <vector>
#include "functions/functions.h"
#include <unordered_map>
using IntArray = std::vector<int>;
using Array = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

// ============== MatMul ==============

Napi::Value DenseForward_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    IntArray weightShape = Vectorize(info[1].As<Napi::Array>());
    std::string activationType = info[2].As<Napi::String>().Utf8Value();
    int pointer = info[5].As<Napi::Number>().Int32Value();
    std::string modelID = info[6].As<Napi::String>().Utf8Value();
    std::string layerID = info[7].As<Napi::String>().Utf8Value();

    int inputSize = weightShape[0];
    int outputSize = weightShape[1];

    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_matmul_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_matmul_activated_output", outputSize);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel matmulKernel = gpu.kernel("matmul");
    cl_kernel activationKernel = gpu.kernel(activationType+"_op"); // activation function is already provided from JS as a lowercase strings. So this will be <activation_string>+"_op" since the forward activations has "_op" suffix

    cl_mem inputTensor = gpu.getOrCreate_Input(modelID, layerID, static_cast<size_t>(inputSize));
    clEnqueueWriteBuffer(queue, inputTensor, CL_FALSE, 0, sizeof(float) * inputSize, input.Data(), 0, nullptr, nullptr);
    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem biases = gpu.getBiases(modelID, pointer);
    cl_mem z_values = gpu.getOrCreate_Z(modelID, layerID, static_cast<size_t>(outputSize));
    cl_mem activated_output = gpu.getOrCreate_ActivationOutput(modelID, layerID, static_cast<size_t>(outputSize));
    

    clSetKernelArg(matmulKernel, 0, sizeof(cl_mem), &inputTensor);
    clSetKernelArg(matmulKernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(matmulKernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(matmulKernel, 3, sizeof(cl_mem), &z_values);
    clSetKernelArg(matmulKernel, 4, sizeof(int), &inputSize);
    clSetKernelArg(matmulKernel, 5, sizeof(int), &outputSize);

    size_t global = size_t(outputSize);

    // run matmul
    clEnqueueNDRangeKernel(queue, matmulKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    // prepare z values vector
    clEnqueueReadBuffer(queue, z_values, CL_TRUE, 0, sizeof(float) * outputSize, z_values_vector.Data(), 0, nullptr, nullptr);

    if (activationType != "softmax") {
        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &activated_output);
        clSetKernelArg(activationKernel, 2, sizeof(int), &outputSize);

        // run activation function
        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        
        clEnqueueReadBuffer(queue, activated_output, CL_TRUE, 0, sizeof(float) * outputSize, activated_output_vector.Data(), 0, nullptr, nullptr);
    }
    else {
        float* data = z_values_vector.Data();

        float max_val = data[0];
        for (int i = 1; i < outputSize; i++) {
            max_val = std::max(max_val, data[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            sum += std::exp(data[i] - max_val);
        }

        clSetKernelArg(activationKernel, 0, sizeof(cl_mem), &z_values);
        clSetKernelArg(activationKernel, 1, sizeof(cl_mem), &activated_output);
        clSetKernelArg(activationKernel, 2, sizeof(float), &max_val);
        clSetKernelArg(activationKernel, 3, sizeof(float), &sum);
        clSetKernelArg(activationKernel, 4, sizeof(int), &outputSize);

        size_t softmax_branch_global_work_size = (size_t)outputSize;

        clEnqueueNDRangeKernel(queue, activationKernel, 1, nullptr, &softmax_branch_global_work_size, nullptr, 0, nullptr, nullptr);

        clEnqueueReadBuffer(queue, activated_output, CL_TRUE, 0, sizeof(float) * outputSize, activated_output_vector.Data(), 0, nullptr, nullptr);
    }
    

    Napi::Object output = Napi::Object::New(env);
    output.Set("z_values", z_values_vector);
    output.Set("activated_output",activated_output_vector);

    return output;
}

Napi::Value DenseForward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    IntArray weightShape = Vectorize(info[1].As<Napi::Array>());
    std::string activationType = info[2].As<Napi::String>().Utf8Value();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[4].As<Napi::Float32Array>();
    std::string modelID = info[6].As<Napi::String>().Utf8Value();
    std::string layerID = info[7].As<Napi::String>().Utf8Value();

    int inputSize = weightShape[0];
    int outputSize = weightShape[1];

    std::unordered_map<std::string, void(*)(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn)> activations = {
        {"linear", LINEAR},
        {"relu", RELU},
        {"sigmoid", SIGMOID},
        {"tanh", TANH},
        {"softmax", SOFTMAX}
    };

    Napi::Float32Array z_values_vector = getOrCreateBuffer(env, modelID, layerID+"_dense_foward_z_values", outputSize);
    Napi::Float32Array activated_output_vector = getOrCreateBuffer(env, modelID, layerID+"_dense_foward_activated_output", outputSize);

    float* in = input.Data();
    float* w = weights.Data();
    float* b = biases.Data();
    float* activated_output = activated_output_vector.Data();
    float* z_values = z_values_vector.Data();

    
    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        z_values[i] = b[i];
    }

    for (int i = 0; i < inputSize; i++) {
        const float v = in[i];
        const int offset = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            z_values[j] += v * w[offset + j];
        }
    }

    activations[activationType](z_values_vector, activated_output_vector);
    
    Napi::Object output = Napi::Object::New(env);
    output.Set("z_values", z_values_vector);
    output.Set("activated_output",activated_output_vector);

    return output;
}

Napi::Value DeltaMatMul_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array delta   = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    int outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weightsArray = info[3].As<Napi::Float32Array>();
    int pointer = info[4].As<Napi::Number>().Int32Value();
    std::string modelID = info[5].As<Napi::String>().Utf8Value();
    std::string layerID = info[6].As<Napi::String>().Utf8Value();

    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_matmul_backward", inputSize);

    auto& gpu = GpuContext::instance();
    cl_context ctx = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel k = gpu.kernel("delta_matmul");

    int deltaSize = delta.ElementLength();

    cl_mem dDelta = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_incoming_delta_matmul", static_cast<size_t>(outputSize));
    clEnqueueWriteBuffer(queue, dDelta, CL_FALSE, 0, sizeof(float) * deltaSize, delta.Data(), 0, nullptr, nullptr);
    cl_mem dOut = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_output_delta_matmul", static_cast<size_t>(inputSize));
    cl_mem dW = gpu.getWeights(modelID, pointer);

    clSetKernelArg(k, 0, sizeof(cl_mem), &dDelta);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dW);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dOut);
    clSetKernelArg(k, 3, sizeof(int),&inputSize);
    clSetKernelArg(k, 4, sizeof(int), &outputSize);

    size_t global = (size_t)inputSize;
    clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, dOut, CL_TRUE, 0, sizeof(float)*inputSize, output.Data(), 0, nullptr, nullptr);


    return output;
}

Napi::Value DeltaMatMul_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array delta = info[0].As<Napi::Float32Array>();
    int inputSize  = info[1].As<Napi::Number>().Int32Value();
    size_t outputSize = info[2].As<Napi::Number>().Int32Value();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    std::string modelID = info[5].As<Napi::String>().Utf8Value();
    std::string layerID = info[6].As<Napi::String>().Utf8Value();

    Napi::Float32Array output = getOrCreateBuffer(env, modelID, layerID+"_matmul_backward", inputSize);

    float* d = delta.Data();
    const float* w = weights.Data();
    float* o = output.Data();
    
    for (int i = 0; i < inputSize; i++) {
        float sum = 0.0f;
        size_t offset = i * outputSize;

        #pragma omp unroll partial(4)        
        for (size_t j = 0; j < outputSize; j++) {
            sum += w[offset + j] * d[j];
        }

        o[i] = sum;
    }

    return output;
}

// ================== Wrappers ====================== //
Napi::Value MatMulWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return DenseForward_GPU(info);
    }

    return DenseForward_CPU(info);
}

Napi::Value DeltaMatMulWrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") {
        return DeltaMatMul_GPU(info);
    }
    return DeltaMatMul_CPU(info);
}


// =================== MODULE EXPORT ===================
void MatMulRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("DesneForward", Napi::Function::New(env, MatMulWrapper));
    exports.Set("DeltaMatMul", Napi::Function::New(env, DeltaMatMulWrapper));
}