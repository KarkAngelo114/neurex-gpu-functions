#include <napi.h>
#include <CL/cl.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <omp.h>
#include <vector>
#include <cmath>

Napi::Value SGD_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array weights_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weight_grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weight_velocity = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biases_array = info[3].As<Napi::Float32Array>();
    Napi::Float32Array biases_grads = info[4].As<Napi::Float32Array>();
    Napi::Float32Array biases_velocity = info[5].As<Napi::Float32Array>();
    float lr = info[6].As<Napi::Number>().FloatValue();
    float momentum = info[7].As<Napi::Number>().FloatValue();
    int batchSize = info[8].As<Napi::Number>().Int32Value();
    bool useBias = info[9].As<Napi::Boolean>().Value();
    std::string modelID = info[10].As<Napi::String>().Utf8Value();
    std::string layerID = info[11].As<Napi::String>().Utf8Value();

    size_t weight_size = weights_array.ElementLength();
    size_t bias_size = weights_array.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel weight_update_kernel = gpu.kernel("sgd");
    cl_kernel bias_update_kernel = gpu.kernel("sgd");

    cl_mem weights = gpu.getWeights(modelID, layerID);
    cl_mem biases = gpu.getBiases(modelID, layerID);
    cl_mem weightVelocity = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_weight_velocity", weight_size);
    cl_mem biasVelocity = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_bias_velocity", bias_size);
    cl_mem weightGrads = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_weight_grads", weight_size);
    cl_mem biasGrads = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_bias_grads", bias_size);
    clEnqueueWriteBuffer(queue, weightGrads, CL_FALSE, 0, sizeof(float) * weight_size, weight_grads.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, biasGrads, CL_FALSE, 0, sizeof(float) * bias_size, biases_grads.Data(), 0, nullptr, nullptr);

    clSetKernelArg(weight_update_kernel, 0, sizeof(cl_mem), &weights);
    clSetKernelArg(weight_update_kernel, 1, sizeof(cl_mem), &weightGrads);
    clSetKernelArg(weight_update_kernel, 2, sizeof(cl_mem), &weightVelocity);
    clSetKernelArg(weight_update_kernel, 3, sizeof(float), &lr);
    clSetKernelArg(weight_update_kernel, 4, sizeof(float), &momentum);
    clSetKernelArg(weight_update_kernel, 5, sizeof(int), &static_cast<int>(weight_size));
    clSetKernelArg(weight_update_kernel, 5, sizeof(int), &batchSize);

    clSetKernelArg(bias_update_kernel, 0, sizeof(cl_mem), &biases);
    clSetKernelArg(bias_update_kernel, 1, sizeof(cl_mem), &biasGrads);
    clSetKernelArg(bias_update_kernel, 2, sizeof(cl_mem), &biasVelocity);
    clSetKernelArg(bias_update_kernel, 3, sizeof(float), &lr);
    clSetKernelArg(bias_update_kernel, 4, sizeof(float), &momentum);
    clSetKernelArg(bias_update_kernel, 5, sizeof(int), &static_cast<int>(bias_size));
    clSetKernelArg(bias_update_kernel, 5, sizeof(int), &batchSize);

    clEnqueueNDRangeKernel(queue, weight_update_kernel, 1, 0, &weight_size, nullptr, 0, nullptr, nullptr);
    
    if (useBias) {
        clEnqueueNDRangeKernel(queue, bias_update_kernel, 1, 0, &bias_size, nullptr, 0, nullptr, nullptr);

        clEnqueueReadBuffer(queue, biases, CL_FALSE, 0, sizeof(float)* bias_size, biases_array.Data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, biasVelocity, CL_FALSE, 0, sizeof(float)* bias_size, biases_velocity.Data(), 0, nullptr, nullptr);
    }

    clEnqueueReadBuffer(queue, weights, CL_FALSE, 0, sizeof(float)* weight_size, weights_array.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, weightVelocity, CL_TRUE, 0, sizeof(float)* weight_size, weight_velocity.Data(), 0, nullptr, nullptr);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights_array);
    output.Set("weightVelocity", weight_velocity);
    output.Set("biases",biases_array);
    output.Set("biasVelocity",biases_velocity);

    return output;
}

Napi::Value SGD_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array weights_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weight_grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weight_velocity = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biases_array = info[3].As<Napi::Float32Array>();
    Napi::Float32Array biases_grads = info[4].As<Napi::Float32Array>();
    Napi::Float32Array biases_velocity = info[5].As<Napi::Float32Array>();
    float lr = info[6].As<Napi::Number>().FloatValue();
    float momentum = info[7].As<Napi::Number>().FloatValue();
    int batchSize = info[8].As<Napi::Number>().Int32Value();
    bool useBias = info[9].As<Napi::Boolean>().Value();

    int weight_size = weights_array.ElementLength();
    int bias_size = weights_array.ElementLength();

    float* w = weights_array.Data();
    float* wg = weight_grads.Data();
    float* wv = weight_velocity.Data();
    float* b = biases_array.Data();
    float* bg = biases_grads.Data();
    float* bv = biases_velocity.Data();

    #pragma omp parallel for
    for (int i = 0; i < weight_size; i++) {
        wv[i] = momentum * wv[i] + (wg[i] / batchSize);
        w[i] -= lr * wv[i];
    }

    if (useBias) {
        #pragma omp parallel for
        for (int i = 0; i < bias_size; i++) {
            bv[i] = momentum * bv[i] + (bg[i] / batchSize);
            b[i] -= lr * bv[i];
        }
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights_array);
    output.Set("weightVelocity", weight_velocity);
    output.Set("biases",biases_array);
    output.Set("biasVelocity",biases_velocity);

    return output;
}

Napi::Value Adam_GPU(const Napi::CallbackInfo& info) {
     Napi::Env env = info.Env();

    Napi::Float32Array weights_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weight_grads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases_array = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biases_grads = info[3].As<Napi::Float32Array>();
    Napi::Float32Array m_weights = info[4].As<Napi::Float32Array>();
    Napi::Float32Array m_biases = info[5].As<Napi::Float32Array>();
    Napi::Float32Array v_weights = info[6].As<Napi::Float32Array>();
    Napi::Float32Array v_biases = info[7].As<Napi::Float32Array>();
    int stateT = info[8].As<Napi::Number>().Int32Value();
    float learning_rate = info[9].As<Napi::Number>().FloatValue();
    float beta1 = info[10].As<Napi::Number>().FloatValue();
    float beta2 = info[11].As<Napi::Number>().FloatValue();
    float epsilon = info[12].As<Napi::Number>().FloatValue();
    int batchSize = info[13].As<Napi::Number>().Int32Value();
    bool useBias = info[14].As<Napi::Boolean>().Value();
    std::string modelID = info[15].As<Napi::String>().Utf8Value();
    std::string layerID = info[16].As<Napi::String>().Utf8Value();

    size_t weightSize = weights_array.ElementLength();
    size_t biasSize = biases_array.ElementLength();
    
    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel weight_update_kernel = gpu.kernel("adam");
    cl_kernel bias_update_kernel = gpu.kernel("adam");

    cl_mem weights = gpu.getWeights(modelID, layerID);
    cl_mem biases = gpu.getBiases(modelID, layerID);
    cl_mem m_weight_state_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_m_weight_state_buffer", weightSize);
    cl_mem v_weight_state_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_v_weight_state_buffer", weightSize);
    cl_mem m_bias_state_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"m_bias_state_buffer", biasSize);
    cl_mem v_bias_state_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"v_bias_state_buffer", biasSize);
    cl_mem weightGrads = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_weight_grads", weight_size);
    cl_mem biasGrads = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_bias_grads", bias_size);
    clEnqueueWriteBuffer(queue, weightGrads, CL_FALSE, 0, sizeof(float) * weight_size, weight_grads.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, biasGrads, CL_FALSE, 0, sizeof(float) * bias_size, biases_grads.Data(), 0, nullptr, nullptr);

    clSetKernelArg(weight_update_kernel, 0, sizeof(cl_mem), &weights);
    clSetKernelArg(weight_update_kernel, 1, sizeof(cl_mem), &weightGrads);
    clSetKernelArg(weight_update_kernel, 2, sizeof(cl_mem), &m_weight_state_buffer);
    clSetKernelArg(weight_update_kernel, 3, sizeof(cl_mem), &v_weight_state_buffer);
    clSetKernelArg(weight_update_kernel, 4, sizeof(int), &stateT);
    clSetKernelArg(weight_update_kernel, 5, sizeof(float), &learning_rate);
    clSetKernelArg(weight_update_kernel, 6, sizeof(float), &beta1);
    clSetKernelArg(weight_update_kernel, 7, sizeof(float), &beta2);
    clSetKernelArg(weight_update_kernel, 8, sizeof(float), &epsilon);
    clSetKernelArg(weight_update_kernel, 9, sizeof(int), &static_cast<int>(weightSize));
    clSetKernelArg(weight_update_kernel, 10, sizeof(int), &batchSize);

    clSetKernelArg(bias_update_kernel, 0, sizeof(cl_mem), &biases);
    clSetKernelArg(bias_update_kernel, 1, sizeof(cl_mem), &biasGrads);
    clSetKernelArg(bias_update_kernel, 2, sizeof(cl_mem), &m_bias_state_buffer);
    clSetKernelArg(bias_update_kernel, 3, sizeof(cl_mem), &v_bias_state_buffer);
    clSetKernelArg(bias_update_kernel, 4, sizeof(int), &stateT);
    clSetKernelArg(bias_update_kernel, 5, sizeof(float), &learning_rate);
    clSetKernelArg(bias_update_kernel, 6, sizeof(float), &beta1);
    clSetKernelArg(bias_update_kernel, 7, sizeof(float), &beta2);
    clSetKernelArg(bias_update_kernel, 8, sizeof(float), &epsilon);
    clSetKernelArg(bias_update_kernel, 9, sizeof(int), &static_cast<int>(biasSize));
    clSetKernelArg(bias_update_kernel, 10, sizeof(int), &batchSize);

    clEnqueueNDRangeKernel(queue, weight_update_kernel, 1, 0, &weightSize, nullptr, 0, nullptr, nullptr);

    if (useBias) {
        clEnqueueNDRangeKernel(queue, bias_update_kernel, 1, 0, &biasSize, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, biases, CL_FALSE, 0, sizeof(float)* biasSize, biases_array.Data(), 0, nullptr, nullptr );
        clEnqueueReadBuffer(queue, m_bias_state_buffer, CL_FALSE, 0, sizeof(float)* biasSize, m_biases.Data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, v_bias_state_buffer, CL_FALSE, 0, sizeof(float)* biasSize, v_biases.Data(), 0, nullptr, nullptr);
    }

    
    clEnqueueReadBuffer(queue, m_weight_state_buffer, CL_FALSE, 0, sizeof(float)* weightSize, m_weights.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, v_weight_state_buffer, CL_FALSE, 0, sizeof(float)* weightSize, v_weights.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, weights, CL_TRUE, 0, sizeof(float)* weightSize, weights_array.Data(), 0, nullptr, nullptr );

    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights_array);
    output.Set("biases", biases_array);
    output.Set("m_weights", m_weights);
    output.Set("m_biases", m_biases);
    output.Set("v_weights", v_weights);
    output.Set("v_biases", v_biases);

    return output;
}

Napi::Value Adam_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array weights = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    Napi::Float32Array m_weights = info[4].As<Napi::Float32Array>();
    Napi::Float32Array m_biases = info[5].As<Napi::Float32Array>();
    Napi::Float32Array v_weights = info[6].As<Napi::Float32Array>();
    Napi::Float32Array v_biases = info[7].As<Napi::Float32Array>();
    int stateT = info[8].As<Napi::Number>().Int32Value();
    float learning_rate = info[9].As<Napi::Number>().FloatValue();
    float beta1 = info[10].As<Napi::Number>().FloatValue();
    float beta2 = info[11].As<Napi::Number>().FloatValue();
    float epsilon = info[12].As<Napi::Number>().FloatValue();
    int batchSize = info[13].As<Napi::Number>().Int32Value();
    bool useBias = info[14].As<Napi::Boolean>().Value();

    int weightSize = weights.ElementLength();
    int biasSize = biases.ElementLength();


    float* w = weights.Data();
    float* wg = weightGrads.Data();
    float* mw = m_weights.Data();
    float* vw = v_weights.Data();
    float* b = biases.Data();
    float* bg = biasGrads.Data();
    float* mb = m_biases.Data();
    float* vb = v_biases.Data();
    

    #pragma omp parallel for
    for (int i = 0; i < weightSize; i++) {
        float grad = (wg[i] / batchSize);

        mw[i] = beta1 * mw[i] + (1 - beta1) * grad;
        vw[i] = beta2 * vw[i] + (1 - beta2) * grad * grad;

        float mHat = mw[i] / (1.0f - pow(beta1, stateT));
        float vHat = vw[i] / (1.0f - pow(beta2, stateT));

        w[i] -= learning_rate * mHat / (sqrt(vHat) + epsilon);
    }

    if (useBias) {
        #pragma omp parallel for
        for (int i = 0; i < biasSize; i++) {
            float grad = (bg[i] / batchSize);

            mb[i] = beta1 * mb[i] + (1 - beta1) * grad;
            vb[i] = beta2 * vb[i] + (1 - beta2) * grad * grad;

            float mHat = mb[i] / (1.0f - pow(beta1, stateT));
            float vHat = vb[i] / (1.0f - pow(beta2, stateT));

            b[i] -= learning_rate * mHat / (sqrt(vHat) + epsilon);
        }
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights);
    output.Set("biases", biases);
    output.Set("m_weights", m_weights);
    output.Set("m_biases", m_biases);
    output.Set("v_weights", v_weights);
    output.Set("v_biases", v_biases);

    return output;
}

Napi::Value RMSProp_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array weights_array = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases_array = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    Napi::Float32Array sqAvg_weights = info[4].As<Napi::Float32Array>();
    Napi::Float32Array sqAvg_biases = info[5].As<Napi::Float32Array>();
    float lr = info[6].As<Napi::Number>().FloatValue();
    float epsilon = info[7].As<Napi::Number>().FloatValue();
    float decayRate = info[8].As<Napi::Number>().FloatValue();
    int batchSize = info[9].As<Napi::Number>().Int32Value();
    bool useBias = info[10].As<Napi::Boolean>().Value();
    std::string modelID = info[11].As<Napi::String>().Utf8Value();
    std::string layerID = info[12].As<Napi::String>().Utf8Value();

    size_t weightSize = weights_array.ElementLength();
    size_t biasSize = biases_array.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel weight_update_kernel = gpu.kernel("rmsprop");
    cl_kernel bias_update_kernel = gpu.kernel("rmsprop");

    cl_mem weights = gpu.getWeights(modelID, layerID);
    cl_mem biases = gpu.getBiases(modelID, layerID);
    cl_mem sqAvg_weight_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_sqAvg_weight_buffer", weightSize);
    cl_mem sqAvg_bias_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_sqAvg_bias_buffer", biasSize);
    cl_mem weightGrads_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_weight_grads", weightSize);
    cl_mem biasGrads_buffer = gpu.getOrCreateOpenCLBuffer(modelID, layerID+"_bias_grads", biasSize);
    clEnqueueWriteBuffer(queue, weightGrads_buffer, CL_FALSE, 0, sizeof(float) * weightSize, weightGrads.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, biasGrads_buffer, CL_FALSE, 0, sizeof(float) * biasSize, biasGrads.Data(), 0, nullptr, nullptr);


    clSetKernelArg(weight_update_kernel, 0, sizeof(cl_mem), &weights);
    clSetKernelArg(weight_update_kernel, 1, sizeof(cl_mem), &weightGrads_buffer);
    clSetKernelArg(weight_update_kernel, 2, sizeof(cl_mem), &sqAvg_weight_buffer);
    clSetKernelArg(weight_update_kernel, 3, sizeof(float), &lr);
    clSetKernelArg(weight_update_kernel, 4, sizeof(float), &epsilon);
    clSetKernelArg(weight_update_kernel, 5, sizeof(float), &decayRate);
    clSetKernelArg(weight_update_kernel, 6, sizeof(int), &static_cast<int>(weightSize));
    clSetKernelArg(weight_update_kernel, 7, sizeof(int), &batchSize);

    clSetKernelArg(bias_update_kernel, 0, sizeof(cl_mem), &biases);
    clSetKernelArg(bias_update_kernel, 1, sizeof(cl_mem), &biasGrads_buffer);
    clSetKernelArg(bias_update_kernel, 2, sizeof(cl_mem), &sqAvg_bias_buffer);
    clSetKernelArg(bias_update_kernel, 3, sizeof(float), &lr);
    clSetKernelArg(bias_update_kernel, 4, sizeof(float), &epsilon);
    clSetKernelArg(bias_update_kernel, 5, sizeof(float), &decayRate);
    clSetKernelArg(bias_update_kernel, 6, sizeof(int), &static_cast<int>(biasSize));
    clSetKernelArg(bias_update_kernel, 7, sizeof(int), &batchSize);

    clEnqueueNDRangeKernel(queue, weight_update_kernel, 1, 0, &weightSize, nullptr, 0, nullptr, nullptr);

    if (useBias) {
        clEnqueueNDRangeKernel(queue, bias_update_kernel, 1, 0, &biasSize, nullptr, 0, nullptr, nullptr);

        clEnqueueReadBuffer(queue, biases, CL_FALSE, 0, sizeof(float)* bias_size, biases_array.Data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, sqAvg_bias_buffer, CL_FALSE, 0, sizeof(float)* bias_size, sqAvg_biases.Data(), 0, nullptr, nullptr);
    }
    
    clEnqueueReadBuffer(queue, sqAvg_weight_buffer, CL_FALSE, 0, sizeof(float)* weightSize, sqAvg_weights.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, weights, CL_TRUE, 0, sizeof(float)* weightSize, weights_array.Data(), 0, nullptr, nullptr);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights_array);
    output.Set("biases", biases_array);
    output.Set("sqAvg_weights", sqAvg_weights);
    output.Set("sqAvg_biases", sqAvg_biases);
    
    return output;
}

Napi::Value RMSProp_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array weights = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    Napi::Float32Array sqAvg_weights = info[4].As<Napi::Float32Array>();
    Napi::Float32Array sqAvg_biases = info[5].As<Napi::Float32Array>();
    float lr = info[6].As<Napi::Number>().FloatValue();
    float epsilon = info[7].As<Napi::Number>().FloatValue();
    float decayRate = info[8].As<Napi::Number>().FloatValue();
    int batchSize = info[9].As<Napi::Number>().Int32Value();
    bool useBias = info[10].As<Napi::Boolean>().Value();

    int weightSize = weights.ElementLength();
    int biasSize = biases.ElementLength();

    float* w = weights.Data();
    float* wg = weightGrads.Data();
    float* sqAvg_w = sqAvg_weights.Data();
    float* b = biases.Data();
    float* bg = biasGrads.Data();
    float* sqAvg_b = sqAvg_biases.Data();

    #pragma omp parallel for
    for (int i = 0; i < weightSize; i++) {
        float g = (wg[i] / batchSize);

        sqAvg_w[i] = decayRate * sqAvg_w[i] + (1 - decayRate) * (g * g);
        w[i] -= (lr / (sqrt(sqAvg_w[i]) + epsilon)) * g;
    }

    if (useBias) {
        #pragma omp parallel for
        for (int i = 0; i < biasSize; i++) {
            float g = (bg[i] / batchSize);

            sqAvg_b[i] = decayRate * sqAvg_b[i] + (1 - decayRate) * (g * g);
            b[i] -= (lr / (sqrt(sqAvg_b[i]) + epsilon)) * g;
        }
    }


    Napi::Object output = Napi::Object::New(env);
    output.Set("weights", weights);
    output.Set("biases", biases);
    output.Set("sqAvg_weights", sqAvg_weights);
    output.Set("sqAvg_biases", sqAvg_biases);
    
    return output;
}

// =============== wrappers ===============

Napi::Value SGD_wrapper(const Napi::CallbackInfo& info) {
    // if (getComputeBackendType() == "opencl") {
    //     return SGD_GPU(info);
    // }

    return SGD_CPU(info);
}

Napi::Value Adam_wrapper(const Napi::CallbackInfo& info) {
    // if (getComputeBackendType() == "opencl") {
    //     return Adam_GPU(info);
    // }

    return Adam_CPU(info);
}

Napi::Value RMSProp_wrapper(const Napi::CallbackInfo& info) {
    // if (getComputeBackendType() == "opencl") {
    //     return RMSProp_GPU(info);
    // }
    return RMSProp_CPU(info);
}

// ======== exports ======== //
void OptimizerInternals(Napi::Env env, Napi::Object exports) {
    exports.Set("SGD", Napi::Function::New(env, SGD_wrapper));
    exports.Set("Adam", Napi::Function::New(env, Adam_wrapper));
    exports.Set("RMSProp", Napi::Function::New(env, RMSProp_wrapper));
}