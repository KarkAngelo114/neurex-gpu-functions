#include <napi.h>
#include <CL/cl.h>
#include <omp.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include "functions/functions.h"
#include <vector>
#include <cmath>
#include <stdexcept>
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

Napi::Value recurrentWeightGradsAccumulation_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // 1. Extract Inputs & Raw Pointers directly
    float* activation_outputs = info[0].As<Napi::Float32Array>().Data();
    float* output = info[4].As<Napi::Float32Array>().Data();

    std::vector<const float*> hiddenStates = ExtractFloat32ArrayPointers(info[2].As<Napi::Array>());
    std::vector<const float*> deltaTs = ExtractFloat32ArrayPointers(info[3].As<Napi::Array>());

    IntArray weightShapeJS = Vectorize(info[5].As<Napi::Array>());
    int sequenceLength = info[6].As<Napi::Number>().Uint32Value();

    // 2. Setup Shape Parameters
    int featureSize = weightShapeJS[0];
    int units = weightShapeJS[1];
    size_t totalInputWeights = featureSize * units;

    // Helper buffer for h_prev when t == 0
    std::vector<float> zero_h_prev(units, 0.0f);

    // 3. Outer Product Loops (Pure C++)
    #pragma omp parallel for
    for (int t = 0; t < sequenceLength; ++t) {
        const float* x_t     = activation_outputs + (t * featureSize);
        const float* delta_t = deltaTs[t];
        
        // Handle t === 0 zero-fill fallback
        const float* h_prev = (t == 0 || hiddenStates[t - 1] == nullptr)  ? zero_h_prev.data() : hiddenStates[t - 1];

        // dL/dW_x += outer(x_t, delta_t)
        for (uint32_t i = 0; i < featureSize; ++i) {
            float xi = x_t[i];
            size_t rowOffset = i * units;

            
            for (uint32_t j = 0; j < units; ++j) {
                output[rowOffset + j] += xi * delta_t[j];
            }
        }

        // dL/dW_h += outer(h_prev, delta_t)
        for (uint32_t i = 0; i < units; ++i) {
            float hi = h_prev[i];
            size_t rowOffset = totalInputWeights + (i * units);

            
            for (uint32_t j = 0; j < units; ++j) {
                output[rowOffset + j] += hi * delta_t[j];
            }
        }
    }

    return info[4];
}

Napi::Value recurrentBiasGradsAccumulation_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array biasGrads_array = info[0].As<Napi::Float32Array>();
    Napi::Array deltaTsJS = info[1].As<Napi::Array>();
    int sequenceLength = info[2].As<Napi::Number>().Int32Value();
    int units = info[3].As<Napi::Number>().Int32Value();

    std::vector<const float*> deltaTs = ExtractFloat32ArrayPointers(deltaTsJS);
    float* biasGrads = biasGrads_array.Data();

    for (int t = 0; t < sequenceLength; t++) {
        const float* delta_time_step = deltaTs[t];

        
        for (int j = 0; j < units; j++) {
            biasGrads[j] += delta_time_step[j];
        }
    }

    return biasGrads_array;
}

Napi::Value accumulateWeightandBiasGradsForTransConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[6].As<Napi::Array>());
    int strides = info[7].As<Napi::Number>().Int32Value();
    std::string modelID = info[8].As<Napi::String>().Utf8Value();
    std::string layerID = info[9].As<Napi::String>().Utf8Value();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int filters = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = padH / 2;
    int padLeft = padW / 2;
    size_t weightGradsSize = weightGrads.ElementLength();
    size_t biasGradsSize = biasGrads.ElementLength();

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulateTransConvWeightAndBiasGrads");

    cl_mem activations = gpu.getInput(modelID, layerID);
    
    cl_mem delta_input = gpu.getDelta(modelID, layerID);
    
    cl_event weightWriteEvent;
    cl_mem weightGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_trans_conv_weight_grads", weightGradsSize);
    clEnqueueWriteBuffer(queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads.Data(), 0, nullptr, &weightWriteEvent);

    cl_event biasWriteEvent;
    cl_mem biasGradsBuffer = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID + "_trans_conv_bias_grads", biasGradsSize);
    clEnqueueWriteBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGrads.Data(), 0, nullptr, &biasWriteEvent);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &activations);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &delta_input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weightGradsBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &biasGradsBuffer);
    clSetKernelArg(kernel, 4, sizeof(int), &iH);
    clSetKernelArg(kernel, 5, sizeof(int), &iW);
    clSetKernelArg(kernel, 6, sizeof(int), &iD);
    clSetKernelArg(kernel, 7, sizeof(int), &oH);
    clSetKernelArg(kernel, 8, sizeof(int), &oW);
    clSetKernelArg(kernel, 9, sizeof(int), &filters);
    clSetKernelArg(kernel, 10, sizeof(int), &kh);
    clSetKernelArg(kernel, 11, sizeof(int), &kw);
    clSetKernelArg(kernel, 12, sizeof(int), &strides);
    clSetKernelArg(kernel, 13, sizeof(int), &padTop);
    clSetKernelArg(kernel, 14, sizeof(int), &padLeft);

    size_t globalSize[3] = {
        static_cast<size_t>(filters),
        static_cast<size_t>(kh),
        static_cast<size_t>(kw) * static_cast<size_t>(iD)
    };

    cl_event writeWaitList[] = { weightWriteEvent, biasWriteEvent };
    cl_event kernelEvent;

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 2, writeWaitList, &kernelEvent);
    clReleaseEvent(weightWriteEvent);
    clReleaseEvent(biasWriteEvent);

    cl_event kernelWaitList[] = { kernelEvent };
    cl_event readEvents[2];

    clEnqueueReadBuffer(queue, weightGradsBuffer, CL_FALSE, 0, sizeof(float) * weightGradsSize, weightGrads.Data(), 1, kernelWaitList, &readEvents[0]);
    clEnqueueReadBuffer(queue, biasGradsBuffer, CL_FALSE, 0, sizeof(float) * biasGradsSize, biasGrads.Data(), 1, kernelWaitList, &readEvents[1]);

    clWaitForEvents(2, readEvents);
    clReleaseEvent(readEvents[0]);
    clReleaseEvent(readEvents[1]);
    clReleaseEvent(kernelEvent);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;
}

Napi::Value accumulateWeightandBiasGradsForTransConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array activation_outputs = info[0].As<Napi::Float32Array>();
    Napi::Float32Array deltas = info[1].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[2].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[3].As<Napi::Float32Array>();
    IntArray inputShape = Vectorize(info[4].As<Napi::Array>());
    IntArray outputShape = Vectorize(info[5].As<Napi::Array>());
    IntArray weightShape = Vectorize(info[6].As<Napi::Array>());
    int strides = info[7].As<Napi::Number>().Int32Value();

    int iH = inputShape[0];
    int iW = inputShape[1];
    int iD = inputShape[2];

    int oH = outputShape[0];
    int oW = outputShape[1];
    int oD = outputShape[2];

    int filters = weightShape[0];
    int kh = weightShape[1];
    int kw = weightShape[2];
    int d = weightShape[3];

    int padH = std::max(0, (iH - 1) * strides + kh - oH);
    int padW = std::max(0, (iW - 1) * strides + kw - oW);
    int padTop = padH / 2;
    int padLeft = padW / 2;

    const float* activationData = activation_outputs.Data();
    const float* deltaData = deltas.Data();
    float* weightGradsData = weightGrads.Data();
    float* biasGradsData = biasGrads.Data();

    for (int iy = 0; iy < iH; iy++) {
        for (int ix = 0; ix < iW; ix++) {
            int inputBase = (iy * iW + ix) * iD;

            for (int ky = 0; ky < kh; ky++) {
                int oy = iy * strides + ky - padTop;
                if (oy < 0 || oy >= oH) continue;

                for (int kx = 0; kx < kw; kx++) {
                    int ox = ix * strides + kx - padLeft;
                    if (ox < 0 || ox >= oW) continue;

                    int deltaBase = (oy * oW + ox) * filters;

                    for (int filter = 0; filter < filters; filter++) {
                        float deltaVal = deltaData[deltaBase + filter];
                        int gradBase = ((filter * kh + ky) * kw + kx) * iD;

                        
                        for (int c = 0; c < iD; c++) {
                            weightGradsData[gradBase + c] += activationData[inputBase + c] * deltaVal;
                        }
                    }
                }
            }
        }
    }

    for (int f = 0; f < filters; f++) {
        float sum = 0.0f;

        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                int idx = (h * oW + w) * filters + f;
                sum += deltas[idx];
            }
        }

        biasGradsData[f] += sum;
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;

}

Napi::Value accumulate_MHA_WeightsAndBiasGrads_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array dQ = info[0].As<Napi::Float32Array>();
    Napi::Float32Array dK = info[1].As<Napi::Float32Array>();
    Napi::Float32Array dV = info[2].As<Napi::Float32Array>();
    Napi::Float32Array dMhaOutput = info[3].As<Napi::Float32Array>();
    Napi::Float32Array mhaOutput = info[4].As<Napi::Float32Array>();
    Napi::Float32Array activationOutputs = info[5].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[6].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[7].As<Napi::Float32Array>();
    int embedDim = info[8].As<Napi::Number>().Int32Value();
    int seqLen = info[9].As<Napi::Number>().Int32Value();

    const float* deltas[] = {dQ.Data(), dK.Data(), dV.Data(), dMhaOutput.Data()};
    const float* inputs[] = {activationOutputs.Data(), activationOutputs.Data(), activationOutputs.Data(), mhaOutput.Data()};
    float* wgOutput = weightGrads.Data();
    float* bgOutput = biasGrads.Data();
    size_t blockSize = static_cast<size_t>(embedDim) * embedDim;
    
    for (int t = 0; t < seqLen; t++) {
        const size_t rowOffset = static_cast<size_t>(t) * embedDim;

        for (int block = 0; block < 4; block++) {
            const size_t blockOffset = static_cast<size_t>(block) * blockSize;

            for (int i = 0; i < embedDim; i++) {
                const float inputValue = inputs[block][rowOffset + i];
                const size_t weightRowOffset = blockOffset + static_cast<size_t>(i) * embedDim;

                #pragma omp unroll partial(4)
                for (int j = 0; j < embedDim; j++) {
                    wgOutput[weightRowOffset + j] += inputValue * deltas[block][rowOffset + j];
                }
            }

            const size_t biasOffset = static_cast<size_t>(block) * embedDim;
            #pragma omp unroll partial(4)
            for (int j = 0; j < embedDim; j++) {
                bgOutput[biasOffset + j] += deltas[block][rowOffset + j];
            }
        }
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;
}

Napi::Value accumulate_MHA_WeightsAndBiasGrads_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array dQ = info[0].As<Napi::Float32Array>();
    Napi::Float32Array dK = info[1].As<Napi::Float32Array>();
    Napi::Float32Array dV = info[2].As<Napi::Float32Array>();
    Napi::Float32Array dMhaOutput = info[3].As<Napi::Float32Array>();
    Napi::Float32Array mhaOutput = info[4].As<Napi::Float32Array>();
    Napi::Float32Array activationOutputs = info[5].As<Napi::Float32Array>();
    Napi::Float32Array weightGrads = info[6].As<Napi::Float32Array>();
    Napi::Float32Array biasGrads = info[7].As<Napi::Float32Array>();
    int embedDim = info[8].As<Napi::Number>().Int32Value();
    int seqLen = info[9].As<Napi::Number>().Int32Value();
    std::string modelID = info[10].As<Napi::String>().Utf8Value();
    std::string layerID = info[11].As<Napi::String>().Utf8Value();

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("accumulate_attention_weight_and_bias_grads");

    size_t tensorBytes = sizeof(float) * static_cast<size_t>(seqLen) * embedDim;
    size_t tensorSize = seqLen * embedDim;
    size_t gradientBytes = sizeof(float) * weightGrads.ElementLength();
    size_t weightGradSize = weightGrads.ElementLength();
    size_t biasGradSize = biasGrads.ElementLength();
    size_t dmhaSize = dMhaOutput.ElementLength();
    size_t biasGradBytes = sizeof(float)* biasGrads.ElementLength();

    cl_mem input = gpu.getInput(modelID, layerID + "_mha_incoming_input");
    cl_mem mha = gpu.getInput(modelID, layerID + "_mha");
    cl_mem q = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_dQ", tensorSize);
    cl_mem k = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_dK", tensorSize);
    cl_mem v = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_dV", tensorSize);
    cl_mem dmha = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_dMha", dmhaSize);
    cl_mem weightGradients = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_mha_weightGrads", weightGradSize);
    cl_mem biasGradients = gpu.getOrCreate_SomethingToWriteOn(modelID, layerID+"_mha_biasGrads", biasGradSize);
    clEnqueueWriteBuffer(queue, weightGradients, CL_FALSE, 0, gradientBytes, weightGrads.Data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, biasGradients, CL_FALSE, 0, biasGradBytes, biasGrads.Data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mha);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &q);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &k);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &v);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &dmha);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &weightGradients);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &biasGradients);
    clSetKernelArg(kernel, 8, sizeof(int), &embedDim);
    clSetKernelArg(kernel, 9, sizeof(int), &seqLen);

    size_t global[3] = {4, static_cast<size_t>(embedDim), static_cast<size_t>(embedDim)};

    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, weightGradients, CL_FALSE, 0, gradientBytes, weightGrads.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, biasGradients, CL_TRUE, 0, biasGradBytes, biasGrads.Data(), 0, nullptr, nullptr);

    Napi::Object output = Napi::Object::New(env);
    output.Set("weightGrads", weightGrads);
    output.Set("biasGrads", biasGrads);

    return output;
}

// =================== wrappers ===================== //

Napi::Value recurrentWeightGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentWeightGradsAccumulation_CPU(info);
}

Napi::Value recurrentBiasGradsAccumulationWrapper(const Napi::CallbackInfo& info) {
    return recurrentBiasGradsAccumulation_CPU(info);
}

Napi::Value accumulateWeightandBiasGradsForTransConv_wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") return accumulateWeightandBiasGradsForTransConv_GPU(info);
    return accumulateWeightandBiasGradsForTransConv_CPU(info);
}

Napi::Value accumulate_MHA_WeightsAndBiasGrads_Wrapper(const Napi::CallbackInfo& info) {
    if (getComputeBackendType() == "opencl") return accumulate_MHA_WeightsAndBiasGrads_GPU(info);
    return accumulate_MHA_WeightsAndBiasGrads_CPU(info);
}

/* ================ module exports ===================*/
void GradientCalculationRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("recurrentWeightGradsAccumulation", Napi::Function::New(env, recurrentWeightGradsAccumulationWrapper));
    exports.Set("recurrentBiasGradsAccumulation", Napi::Function::New(env, recurrentBiasGradsAccumulationWrapper));
    exports.Set("accumulateWeightandBiasGradsForTransConv", Napi::Function::New(env, accumulateWeightandBiasGradsForTransConv_wrapper));
    exports.Set("accumulate_MHA_WeightsAndBiasGrads", Napi::Function::New(env, accumulate_MHA_WeightsAndBiasGrads_Wrapper));
}