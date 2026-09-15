#include <napi.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "functions/functions.h"
#include "gpu/gpu_context.h"
#include "globals/globals.h"

using FloatArray = std::vector<float>;
using String = std::string;

// ===================== shared math helpers (mirror index.js primitives) =====================

// mirrors JS dotProduct(arr1, arr2, inputSize, outputSize):
// row-vector [inputSize] times row-major matrix [inputSize x outputSize] -> [outputSize]
static void _helper_dotProduct(const float* arr1, const float* mat, int inputSize, int outputSize, float* output) {
    #pragma omp unroll partial(4)
    for (int j = 0; j < outputSize; j++) {
        output[j] = 0.0f;
    }


    for (int i = 0; i < inputSize; i++) {
        float v = arr1[i];
        int rowStart = i * outputSize;

        #pragma omp unroll partial(4)
        for (int j = 0; j < outputSize; j++) {
            output[j] += v * mat[rowStart + j];
        }
    }
}

// mirrors JS scale(input, scalingValue): in-place divide
static void _helper_scale(float* data, int length, float scalingValue) {
    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        data[i] /= scalingValue;
    }
}

// mirrors JS Softmax(arr): numerically-stable softmax over one row
static void _helper_softmax(const float* input, int length, float* output) {
    float maxVal = input[0];
    for (int i = 1; i < length; i++) {
        maxVal = std::max(maxVal, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = std::exp(input[i] - maxVal);
        sum += output[i];
    }

    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// mirrors JS DSoftmax(arr1, arr2): arr1 = softmax output row (S), arr2 = incoming dS row
static void _helper_dsoftmax(const float* s, const float* dS, int length, float* output) {
    float dot = 0.0f;
    for (int i = 0; i < length; i++) {
        dot += dS[i] * s[i];
    }

    #pragma omp unroll partial(4)
    for (int i = 0; i < length; i++) {
        output[i] = s[i] * (dS[i] - dot);
    }
}

// mirrors JS transpose2D(matrix, rows, cols): flattened [rows x cols] -> flattened [cols x rows]
static void _helper_transpose2D(const float* matrix, int rows, int cols, float* output) {
    for (int r = 0; r < rows; r++) {

        #pragma omp unroll partial(4)
        for (int c = 0; c < cols; c++) {
            output[c * rows + r] = matrix[r * cols + c];
        }
    }
}

void _helper_matMul(const float* input, int inputSize, int outputSize, const float* weights, const float* biases, float* output) {
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



Napi::Value projectToQKV_GPU(const Napi::CallbackInfo& info) {
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
    int pointer = info[9].As<Napi::Number>().Int32Value();
    std::string modelID = info[10].As<Napi::String>().Utf8Value();

    Napi::Float32Array Q = Napi::Float32Array::New(env, size);
    Napi::Float32Array K = Napi::Float32Array::New(env, size);
    Napi::Float32Array V = Napi::Float32Array::New(env, size);

    auto& gpu = GpuContext::instance();
    cl_command_queue queue = gpu.queue();
    cl_context context = gpu.context();
    cl_kernel kernel = gpu.kernel("projectQKV");


    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* inputTensor.ElementLength(), inputTensor.Data(), nullptr);
    cl_mem weights = gpu.getWeights(modelID, pointer);
    cl_mem biases = gpu.getBiases(modelID, pointer);
    cl_mem _Q = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);
    cl_mem _K = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);
    cl_mem _V = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)* size, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_Q);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &_K);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &_V);
    clSetKernelArg(kernel, 6, sizeof(int), &embeddingDim);
    clSetKernelArg(kernel, 7, sizeof(int), &sequenceLen);

    size_t global = (size_t)sequenceLen;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);

    clEnqueueReadBuffer( queue, _Q, CL_TRUE, 0, sizeof(float) * size, Q.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer( queue, _K, CL_TRUE, 0, sizeof(float) * size, K.Data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer( queue, _V, CL_TRUE, 0, sizeof(float) * size, V.Data(), 0, nullptr, nullptr);

    clReleaseMemObject(input);
    clReleaseMemObject(_Q);
    clReleaseMemObject(_K);
    clReleaseMemObject(_V);

    Napi::Object output = Napi::Object::New(env);
    output.Set("Q", Q);
    output.Set("K", K);
    output.Set("V", V);

    return output;
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

Napi::Value CoreAttention_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[2].As<Napi::Float32Array>();
    int embedDim = info[3].As<Napi::Number>().Int32Value();
    int seqLen = info[4].As<Napi::Number>().Int32Value();
    float dkRoot = info[5].As<Napi::Number>().FloatValue();

    // int pointer = info[6].As<Napi::Number>().Int32Value(); // use only on GPU side
    // String ModelID = info[7].As<Napi::String>().Utf8Value(); // use only on GPU side

    /**
     * based on the old JS implementation of `simpleAttention` core mechanism, we:
     * 1. unpack the weights and biases to QkV
     * 2. project the input to QKV to get the Q_o, K_o, V_o
     * 3. Transposed the K_o
     * 4. Get the attention scores of Q_o rows and transposed_K by doing a dot product per sequence
     * 5. scale all scores by dkRoot
     * 6. apply softmax each scaled scores by row
     * 7. perform dot product with softmaxed scores with V_o per sequence
     */

    const float* input = inputTensor.Data();
    const float* w = weights.Data();
    const float* b = biases.Data();

    // 1. unpack Q/K/V weights and biases (matches unpackQKVO with includeOutputProjectionParameters=false)
    int matrixSize = embedDim * embedDim;
    const float* Q_w = w;
    const float* K_w = w + matrixSize;
    const float* V_w = w + matrixSize * 2;
    const float* Q_b = b;
    const float* K_b = b + embedDim;
    const float* V_b = b + embedDim * 2;

    // outputs we need to return for caching (X, Q, K, V, S) plus the layer output
    Napi::Float32Array Q = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array K = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array V = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array S = Napi::Float32Array::New(env, seqLen * seqLen);
    Napi::Float32Array output = Napi::Float32Array::New(env, seqLen * embedDim);

    float* Q_ptr = Q.Data();
    float* K_ptr = K.Data();
    float* V_ptr = V.Data();
    float* S_ptr = S.Data();
    float* out_ptr = output.Data();

    // 2. project input to Q, K, V — parallel over tokens
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        int offset = t * embedDim;
        const float* tokenVec = input + offset;

        _helper_matMul(tokenVec, embedDim, embedDim, Q_w, Q_b, Q_ptr + offset);
        _helper_matMul(tokenVec, embedDim, embedDim, K_w, K_b, K_ptr + offset);
        _helper_matMul(tokenVec, embedDim, embedDim, V_w, V_b, V_ptr + offset);
    }

    // 3. transpose K -> [embedDim x seqLen]
    FloatArray transposeK(embedDim * seqLen);
    _helper_transpose2D(K_ptr, seqLen, embedDim, transposeK.data());

    // 4 to 5. scores = Q . transpose(K), then scale by dkRoot — done per row, parallel over tokens
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* Qrow = Q_ptr + t * embedDim;
        float* scoreRow = S_ptr + t * seqLen; // reuse S buffer as scratch for raw scores before softmax
        _helper_dotProduct(Qrow, transposeK.data(), embedDim, seqLen, scoreRow);
        _helper_scale(scoreRow, seqLen, dkRoot);
    }

    // 6. softmax each row in place (S_ptr currently holds scaled scores; softmax overwrites with final S)
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        float* row = S_ptr + t * seqLen;
        _helper_softmax(row, seqLen, row);
    }

    // 7. output = softmax(scores) * V — parallel over tokens
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* srow = S_ptr + t * seqLen;
        _helper_dotProduct(srow, V_ptr, seqLen, embedDim, out_ptr + t * embedDim);
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("X", inputTensor);
    result.Set("Q", Q);
    result.Set("K", K);
    result.Set("V", V);
    result.Set("S", S);
    result.Set("output", output);
    return result;
}


Napi::Value CoreAttentionBackward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array incomingDeltaTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array Qtensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Ktensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array Vtensor = info[3].As<Napi::Float32Array>();
    Napi::Float32Array storedStensor = info[4].As<Napi::Float32Array>();
    Napi::Float32Array weightsTensor = info[5].As<Napi::Float32Array>();
    int embedDim = info[6].As<Napi::Number>().Int32Value();
    int seqLen = info[7].As<Napi::Number>().Int32Value();
    float dkRoot = info[8].As<Napi::Number>().FloatValue();

    // int pointer = info[9].As<Napi::Number>().Int32Value(); // GPU side only
    // String modelID = info[10].As<Napi::String>().Utf8Value(); // GPU side only

    const float* incomingDelta = incomingDeltaTensor.Data();
    const float* Q = Qtensor.Data();
    const float* K = Ktensor.Data();
    const float* V = Vtensor.Data();
    const float* storedS = storedStensor.Data();
    const float* w = weightsTensor.Data();

    // unpack Q/K/V weights only (matches unpackQKVO(weights, null, null, null, embedDim))
    int matrixSize = embedDim * embedDim;
    const float* Q_w = w;
    const float* K_w = w + matrixSize;
    const float* V_w = w + matrixSize * 2;

    Napi::Float32Array dQ = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dK = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dV = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dX = Napi::Float32Array::New(env, seqLen * embedDim);

    float* dQ_ptr = dQ.Data();
    float* dK_ptr = dK.Data();
    float* dV_ptr = dV.Data();
    float* dX_ptr = dX.Data();

    // transpose_V = V^T -> [embedDim x seqLen]
    FloatArray transposeV(embedDim * seqLen);
    _helper_transpose2D(V, seqLen, embedDim, transposeV.data());

    // dS = incomingDelta * transpose(V), row per token
    FloatArray dS(seqLen * seqLen);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* deltaRow = incomingDelta + t * embedDim;
        _helper_dotProduct(deltaRow, transposeV.data(), embedDim, seqLen, dS.data() + t * seqLen);
    }

    // transpose_S = storedS^T -> [seqLen x seqLen], then dV = transpose_S . incomingDelta per column-as-row
    FloatArray transposeS(seqLen * seqLen);
    _helper_transpose2D(storedS, seqLen, seqLen, transposeS.data());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < seqLen; k++) {
        const float* sCol = transposeS.data() + k * seqLen;
        _helper_dotProduct(sCol, incomingDelta, seqLen, embedDim, dV_ptr + k * embedDim);
    }

    // apply softmax derivative (Jacobian-vector product) row by row: dScaled = DSoftmax(storedS_row, dS_row)
    FloatArray dScaled(seqLen * seqLen);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* sRow = storedS + t * seqLen;
        const float* dSRow = dS.data() + t * seqLen;
        _helper_dsoftmax(sRow, dSRow, seqLen, dScaled.data() + t * seqLen);
    }

    // dScores = scale(dScaled, dkRoot) — in place
    _helper_scale(dScaled.data(), seqLen * seqLen, dkRoot);

    // dQ = dScores . K, per row
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* dScoreRow = dScaled.data() + t * seqLen;
        _helper_dotProduct(dScoreRow, K, seqLen, embedDim, dQ_ptr + t * embedDim);
    }

    // dK = transpose(dScores) . Q, per row (transpose_dScores columns become rows)
    FloatArray transposeDScores(seqLen * seqLen);
    _helper_transpose2D(dScaled.data(), seqLen, seqLen, transposeDScores.data());

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < seqLen; k++) {
        const float* col = transposeDScores.data() + k * seqLen;
        _helper_dotProduct(col, Q, seqLen, embedDim, dK_ptr + k * embedDim);
    }

    // project dQ/dK/dV back through the transposed weight matrices and sum into dX
    FloatArray transposeQw(embedDim * embedDim);
    FloatArray transposeKw(embedDim * embedDim);
    FloatArray transposeVw(embedDim * embedDim);
    _helper_transpose2D(Q_w, embedDim, embedDim, transposeQw.data());
    _helper_transpose2D(K_w, embedDim, embedDim, transposeKw.data());
    _helper_transpose2D(V_w, embedDim, embedDim, transposeVw.data());

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* dQrow = dQ_ptr + t * embedDim;
        const float* dKrow = dK_ptr + t * embedDim;
        const float* dVrow = dV_ptr + t * embedDim;

        FloatArray fromQ(embedDim), fromK(embedDim), fromV(embedDim);
        _helper_dotProduct(dQrow, transposeQw.data(), embedDim, embedDim, fromQ.data());
        _helper_dotProduct(dKrow, transposeKw.data(), embedDim, embedDim, fromK.data());
        _helper_dotProduct(dVrow, transposeVw.data(), embedDim, embedDim, fromV.data());

        float* dXrow = dX_ptr + t * embedDim;
        for (int d = 0; d < embedDim; d++) {
            dXrow[d] = fromQ[d] + fromK[d] + fromV[d];
        }
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("dQ", dQ);
    result.Set("dK", dK);
    result.Set("dV", dV);
    result.Set("dX", dX);
    return result;
}

// ============ wrappers ==================
Napi::Value projectToQKV_Wrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return projectToQKV_GPU(info);
    }
    return projectToQKV_CPU(info);
}

Napi::Value CoreAttention_Wrapper(const Napi::CallbackInfo& info) {
    return CoreAttention_CPU(info);
}

Napi::Value CoreAttentionBackward_Wrapper(const Napi::CallbackInfo& info) {
    return CoreAttentionBackward_CPU(info);
}

void attentionFunctions(Napi::Env env, Napi::Object exports) {
    exports.Set("projectToQKV", Napi::Function::New(env, projectToQKV_Wrapper));
    exports.Set("CoreAttention", Napi::Function::New(env, CoreAttention_Wrapper));
    exports.Set("CoreAttentionBackward", Napi::Function::New(env, CoreAttentionBackward_Wrapper));
}