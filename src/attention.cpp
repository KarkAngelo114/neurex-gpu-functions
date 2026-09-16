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

// mirrors the per-head slicing loop used in both CoreMultiHeadAttention and its backward pass:
// pulls out this head's [seqLen, headDim] slice from a [seqLen, embedDim] buffer (strided by headOffset)
static void _helper_extractHeadSlice(const float* full, int seqLen, int embedDim, int headDim, int headOffset, float* output) {
    for (int t = 0; t < seqLen; t++) {
        const float* src = full + t * embedDim + headOffset;
        float* dst = output + t * headDim;
        for (int d = 0; d < headDim; d++) {
            dst[d] = src[d];
        }
    }
}

// inverse of _helper_extractHeadSlice: writes a [seqLen, headDim] head buffer back into
// its headOffset-th slot of a [seqLen, embedDim] buffer
static void _helper_writeHeadSlice(const float* headBuf, int seqLen, int embedDim, int headDim, int headOffset, float* full) {
    for (int t = 0; t < seqLen; t++) {
        const float* src = headBuf + t * headDim;
        float* dst = full + t * embedDim + headOffset;
        for (int d = 0; d < headDim; d++) {
            dst[d] = src[d];
        }
    }
}

// applies causal masking to a [seqLen x seqLen] score/grad matrix in place;
// maskValue is -1e9 for forward scores (pre-softmax), 0 for backward dScores (post-softmax-derivative)
static void _helper_applyCausalMask(float* scores, int seqLen, float maskValue) {
    for (int i = 0; i < seqLen; i++) {
        for (int j = i + 1; j < seqLen; j++) {
            scores[i * seqLen + j] = maskValue;
        }
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

Napi::Value CoreMultiHeadAttention_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array inputTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weights = info[1].As<Napi::Float32Array>();
    Napi::Float32Array biases = info[2].As<Napi::Float32Array>();
    int embedDim = info[3].As<Napi::Number>().Int32Value();
    int seqLen = info[4].As<Napi::Number>().Int32Value();
    int numHeads = info[5].As<Napi::Number>().Int32Value();
    int headDim = info[6].As<Napi::Number>().Int32Value();
    float dkRoot = info[7].As<Napi::Number>().FloatValue();
    bool useCasualMasking = info[8].As<Napi::Boolean>().Value();

    // int pointer = info[9].As<Napi::Number>().Int32Value(); // use only on GPU side
    // String ModelID = info[10].As<Napi::String>().Utf8Value(); // use only on GPU side

    const float* input = inputTensor.Data();
    const float* w = weights.Data();
    const float* b = biases.Data();

    // 1. unpack Q/K/V/O weights and biases (matches unpackQKVO with includeOutputProjectionParameters=true)
    int matrixSize = embedDim * embedDim;
    const float* Q_w = w;
    const float* K_w = w + matrixSize;
    const float* V_w = w + matrixSize * 2;
    const float* O_w = w + matrixSize * 3;
    const float* Q_b = b;
    const float* K_b = b + embedDim;
    const float* V_b = b + embedDim * 2;
    const float* O_b = b + embedDim * 3;

    // outputs needed for caching (X, Q, K, V, S_perHead) plus the layer's final output
    Napi::Float32Array Q = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array K = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array V = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array mhaOutput = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array S_perHead = Napi::Float32Array::New(env, numHeads * seqLen * seqLen);
    Napi::Float32Array finalOutput = Napi::Float32Array::New(env, seqLen * embedDim);

    float* Q_ptr = Q.Data();
    float* K_ptr = K.Data();
    float* V_ptr = V.Data();
    float* mhaOut_ptr = mhaOutput.Data();
    float* S_ptr = S_perHead.Data();
    float* final_ptr = finalOutput.Data();

    // 2. project input to Q, K, V [seqLen, embedDim] — parallel over tokens
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        int offset = t * embedDim;
        const float* tokenVec = input + offset;

        _helper_matMul(tokenVec, embedDim, embedDim, Q_w, Q_b, Q_ptr + offset);
        _helper_matMul(tokenVec, embedDim, embedDim, K_w, K_b, K_ptr + offset);
        _helper_matMul(tokenVec, embedDim, embedDim, V_w, V_b, V_ptr + offset);
    }

    int scoresPerHead = seqLen * seqLen;

    // 3. process each head independently — parallel over heads, each thread owns its own scratch buffers
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < numHeads; h++) {
        int headOffset = h * headDim;

        // extract this head's Q_h, K_h, V_h slices [seqLen, headDim]
        FloatArray Q_h(seqLen * headDim);
        FloatArray K_h(seqLen * headDim);
        FloatArray V_h(seqLen * headDim);
        _helper_extractHeadSlice(Q_ptr, seqLen, embedDim, headDim, headOffset, Q_h.data());
        _helper_extractHeadSlice(K_ptr, seqLen, embedDim, headDim, headOffset, K_h.data());
        _helper_extractHeadSlice(V_ptr, seqLen, embedDim, headDim, headOffset, V_h.data());

        // scores = Q_h * transpose(K_h)
        FloatArray transposeK_h(headDim * seqLen);
        _helper_transpose2D(K_h.data(), seqLen, headDim, transposeK_h.data());

        FloatArray scores(scoresPerHead);
        for (int t = 0; t < seqLen; t++) {
            _helper_dotProduct(Q_h.data() + t * headDim, transposeK_h.data(), headDim, seqLen, scores.data() + t * seqLen);
        }

        // causal masking happens BEFORE scaling, matching the JS order
        if (useCasualMasking) {
            _helper_applyCausalMask(scores.data(), seqLen, -1e9f);
        }

        // scale, then softmax per row
        _helper_scale(scores.data(), scoresPerHead, dkRoot);

        float* headS = S_ptr + h * scoresPerHead; // this head's slot in the flattened S_perHead buffer
        for (int t = 0; t < seqLen; t++) {
            _helper_softmax(scores.data() + t * seqLen, seqLen, headS + t * seqLen);
        }

        // headOut = softmax(scores) * V_h, then scatter back into mhaOutput at this head's column offset
        FloatArray headOut(seqLen * headDim);
        for (int t = 0; t < seqLen; t++) {
            _helper_dotProduct(headS + t * seqLen, V_h.data(), seqLen, headDim, headOut.data() + t * headDim);
        }
        _helper_writeHeadSlice(headOut.data(), seqLen, embedDim, headDim, headOffset, mhaOut_ptr);
    }

    // 4. final linear projection (W_O): finalOutput = mhaOutput * O_weights + O_bias, per token
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* mhaRow = mhaOut_ptr + t * embedDim;
        _helper_matMul(mhaRow, embedDim, embedDim, O_w, O_b, final_ptr + t * embedDim);
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("X", inputTensor);
    result.Set("Q", Q);
    result.Set("K", K);
    result.Set("V", V);
    result.Set("mhaOutput", mhaOutput);
    result.Set("S_perHead", S_perHead);
    result.Set("finalOutput", finalOutput);
    return result;
}

Napi::Value CoreMultiHeadAttentionBackward_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array incomingDeltaTensor = info[0].As<Napi::Float32Array>();
    Napi::Float32Array weightsTensor = info[1].As<Napi::Float32Array>();
    Napi::Float32Array Qtensor = info[2].As<Napi::Float32Array>();
    Napi::Float32Array Ktensor = info[3].As<Napi::Float32Array>();
    Napi::Float32Array Vtensor = info[4].As<Napi::Float32Array>();
    Napi::Float32Array S_perHeadTensor = info[5].As<Napi::Float32Array>();
    int embedDim = info[6].As<Napi::Number>().Int32Value();
    int seqLen = info[7].As<Napi::Number>().Int32Value();
    int numHeads = info[8].As<Napi::Number>().Int32Value();
    int headDim = info[9].As<Napi::Number>().Int32Value();
    float dkRoot = info[10].As<Napi::Number>().FloatValue();
    bool useCasualMasking = info[11].As<Napi::Boolean>().Value();

    // int pointer = info[12].As<Napi::Number>().Int32Value(); // GPU side only
    // String modelID = info[13].As<Napi::String>().Utf8Value(); // GPU side only

    const float* incomingDelta = incomingDeltaTensor.Data();
    const float* w = weightsTensor.Data();
    const float* Q = Qtensor.Data();
    const float* K = Ktensor.Data();
    const float* V = Vtensor.Data();
    const float* S_perHead = S_perHeadTensor.Data();

    // unpack Q/K/V/O weights (matches unpackQKVO(weights, null, null, null, embedDim, true))
    int matrixSize = embedDim * embedDim;
    const float* Q_w = w;
    const float* K_w = w + matrixSize;
    const float* V_w = w + matrixSize * 2;
    const float* O_w = w + matrixSize * 3;

    Napi::Float32Array dQ = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dK = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dV = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dMhaOutput = Napi::Float32Array::New(env, seqLen * embedDim);
    Napi::Float32Array dX = Napi::Float32Array::New(env, seqLen * embedDim);

    float* dQ_ptr = dQ.Data();
    float* dK_ptr = dK.Data();
    float* dV_ptr = dV.Data();
    float* dMha_ptr = dMhaOutput.Data();
    float* dX_ptr = dX.Data();

    // dMhaOutput = incomingDelta . transpose(O_weights), per token
    FloatArray transposeO(embedDim * embedDim);
    _helper_transpose2D(O_w, embedDim, embedDim, transposeO.data());

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seqLen; t++) {
        const float* deltaRow = incomingDelta + t * embedDim;
        _helper_dotProduct(deltaRow, transposeO.data(), embedDim, embedDim, dMha_ptr + t * embedDim);
    }

    int scoresPerHead = seqLen * seqLen;

    // per-head backward — parallel over heads, each thread owns its own scratch buffers
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < numHeads; h++) {
        int headOffset = h * headDim;
        const float* S = S_perHead + h * scoresPerHead;

        // slice this head's Q_h, K_h, V_h and its share of dMhaOutput
        FloatArray Q_h(seqLen * headDim);
        FloatArray K_h(seqLen * headDim);
        FloatArray V_h(seqLen * headDim);
        FloatArray dHeadOut(seqLen * headDim);
        _helper_extractHeadSlice(Q, seqLen, embedDim, headDim, headOffset, Q_h.data());
        _helper_extractHeadSlice(K, seqLen, embedDim, headDim, headOffset, K_h.data());
        _helper_extractHeadSlice(V, seqLen, embedDim, headDim, headOffset, V_h.data());
        _helper_extractHeadSlice(dMha_ptr, seqLen, embedDim, headDim, headOffset, dHeadOut.data());

        // dS = dHeadOut * transpose(V_h), per token
        FloatArray transposeV_h(headDim * seqLen);
        _helper_transpose2D(V_h.data(), seqLen, headDim, transposeV_h.data());

        FloatArray dS(scoresPerHead);
        for (int t = 0; t < seqLen; t++) {
            _helper_dotProduct(dHeadOut.data() + t * headDim, transposeV_h.data(), headDim, seqLen, dS.data() + t * seqLen);
        }

        // dV_h = transpose(S) * dHeadOut, per column-as-row
        FloatArray transposeS(scoresPerHead);
        _helper_transpose2D(S, seqLen, seqLen, transposeS.data());

        FloatArray dV_h(seqLen * headDim);
        for (int k = 0; k < seqLen; k++) {
            _helper_dotProduct(transposeS.data() + k * seqLen, dHeadOut.data(), seqLen, headDim, dV_h.data() + k * headDim);
        }

        // softmax derivative row by row: dScaled = DSoftmax(S_row, dS_row)
        FloatArray dScaled(scoresPerHead);
        for (int t = 0; t < seqLen; t++) {
            _helper_dsoftmax(S + t * seqLen, dS.data() + t * seqLen, seqLen, dScaled.data() + t * seqLen);
        }

        // scale by dkRoot
        _helper_scale(dScaled.data(), scoresPerHead, dkRoot);

        // causal masking on dScores happens AFTER scaling, matching the JS order — masked positions get zero gradient
        if (useCasualMasking) {
            _helper_applyCausalMask(dScaled.data(), seqLen, 0.0f);
        }

        // dQ_h = dScores * K_h, per row
        FloatArray dQ_h(seqLen * headDim);
        for (int t = 0; t < seqLen; t++) {
            _helper_dotProduct(dScaled.data() + t * seqLen, K_h.data(), seqLen, headDim, dQ_h.data() + t * headDim);
        }

        // dK_h = transpose(dScores) * Q_h, per column-as-row
        FloatArray transposeDScores(scoresPerHead);
        _helper_transpose2D(dScaled.data(), seqLen, seqLen, transposeDScores.data());

        FloatArray dK_h(seqLen * headDim);
        for (int k = 0; k < seqLen; k++) {
            _helper_dotProduct(transposeDScores.data() + k * seqLen, Q_h.data(), seqLen, headDim, dK_h.data() + k * headDim);
        }

        // scatter this head's dQ_h/dK_h/dV_h back into the full embedDim-wide dQ/dK/dV buffers
        _helper_writeHeadSlice(dQ_h.data(), seqLen, embedDim, headDim, headOffset, dQ_ptr);
        _helper_writeHeadSlice(dK_h.data(), seqLen, embedDim, headDim, headOffset, dK_ptr);
        _helper_writeHeadSlice(dV_h.data(), seqLen, embedDim, headDim, headOffset, dV_ptr);
    }

    // project dQ/dK/dV back through the transposed Q/K/V weight matrices and sum into dX
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
    result.Set("dMhaOutput", dMhaOutput);
    result.Set("dX", dX);
    return result;
}

// ============ wrappers ==================

Napi::Value CoreAttention_Wrapper(const Napi::CallbackInfo& info) {
    return CoreAttention_CPU(info);
}

Napi::Value CoreAttentionBackward_Wrapper(const Napi::CallbackInfo& info) {
    return CoreAttentionBackward_CPU(info);
}

Napi::Value CoreMultiHead_wrapper(const Napi::CallbackInfo& info) {
    return CoreMultiHeadAttention_CPU(info);
}

Napi::Value CoreMultiHeadBackward_wrapper(const Napi::CallbackInfo& info) {
    return CoreMultiHeadAttentionBackward_CPU(info);
}

void attentionFunctions(Napi::Env env, Napi::Object exports) {
    exports.Set("CoreAttention", Napi::Function::New(env, CoreAttention_Wrapper));
    exports.Set("CoreAttentionBackward", Napi::Function::New(env, CoreAttentionBackward_Wrapper));
    exports.Set("CoreMultiHeadAttention", Napi::Function::New(env, CoreMultiHead_wrapper));
    exports.Set("CoreMultiHeadAttentionBackward", Napi::Function::New(env, CoreMultiHeadBackward_wrapper));
}