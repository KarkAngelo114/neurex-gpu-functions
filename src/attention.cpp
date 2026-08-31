#include <napi.h>
#include <omp.h>

void _helper_matMul(
    const float* input, 
    int inputSize, 
    int outputSize, 
    const float* weights, 
    const float* biases, 
    float* output
) {
    #pragma omp unroll partial(4)
    for (int j = 0; j < outputSize; j++) {
        output[j] = biases[j];
    }

    
    for (int i = 0; i < inputSize; i++) {
        float inputVal = input[i];
        int rowStart = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            output[j] += inputVal * weights[rowStart + j];
        }
    }
}

Napi::Value projectToQKV_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array Q_weights_tensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Q_bias_tensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array K_weights_tensor = info[3].As<Napi::Float32Array>();
    Napi::Float32Array K_bias_tensor = info[4].As<Napi::Float32Array>();
    Napi::Float32Array V_weights_tensor = info[5].As<Napi::Float32Array>();
    Napi::Float32Array V_bias_tensor = info[6].As<Napi::Float32Array>();
    int embeddingDim = info[7].As<Napi::Number>().Int32Value();
    int sequenceLen = info[8].As<Napi::Number>().Int32Value();
    int size = embeddingDim * sequenceLen;

    Napi::Float32Array Q = Napi::Float32Array::New(env, size);
    Napi::Float32Array K = Napi::Float32Array::New(env, size);
    Napi::Float32Array V = Napi::Float32Array::New(env, size);

    // Extract raw pointers for C++ CPU operations
    const float* inputPtr = inputTensor.Data();
    const float* Q_w = Q_weights_tensor.Data();
    const float* Q_b = Q_bias_tensor.Data();
    const float* K_w = K_weights_tensor.Data();
    const float* K_b = K_bias_tensor.Data();
    const float* V_w = V_weights_tensor.Data();
    const float* V_b = V_bias_tensor.Data();

    float* Q_ptr = Q.Data();
    float* K_ptr = K.Data();
    float* V_ptr = V.Data();

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < sequenceLen; t++) {
        int offset = t * embeddingDim;

        const float* tokenVec = inputPtr + offset;

        // Perform linear projection onto Q, K, and V
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, Q_w, Q_b, Q_ptr + offset);
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, K_w, K_b, K_ptr + offset);
        _helper_matMul(tokenVec, embeddingDim, embeddingDim, V_w, V_b, V_ptr + offset);
    }

    Napi::Object output = Napi::Object::New(env);
    output.Set("Q", Q);
    output.Set("K", K);
    output.Set("V", V);

    return output;
}


Napi::Value projectToQKV_Wrapper(const Napi::CallbackInfo& info) {
    return projectToQKV_CPU(info);
}


void attentionFunctions(Napi::Env env, Napi::Object exports) {
    exports.Set("projectToQKV", Napi::Function::New(env, projectToQKV_Wrapper));
}