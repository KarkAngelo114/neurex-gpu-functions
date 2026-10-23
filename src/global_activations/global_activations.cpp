// This module is for setting the activation functions for functions that runs on CPU branch, particularly if they perform bulk operations
#include <napi.h>
#include <omp.h>
#include <algorithm>
#include <cmath>


void LINEAR(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* z_values = referenceDataBuffer.Data();
    float* activated_output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        activated_output[i] = z_values[i];
    }
}

void RELU(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* z_values = referenceDataBuffer.Data();
    float* activated_output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        activated_output[i] = std::max(0.0f, z_values[i]);
    }
}

void SIGMOID(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* z_values = referenceDataBuffer.Data();
    float* activated_output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        activated_output[i] = 1.0f / (1.0f + exp(-z_values[i]));
    }
}

void TANH(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* z_values = referenceDataBuffer.Data();
    float* activated_output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        activated_output[i] = std::tanh(z_values[i]);
    }
}

void SOFTMAX(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {

    float* z_values = referenceDataBuffer.Data();
    float* activated_output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    float maxValue = z_values[0];
    #pragma omp unroll partial(4)
    for (int j = 1; j < outputSize; j++) {
        maxValue = std::max(maxValue, z_values[j]);
    }

    float sum = 0.0f;
    #pragma omp unroll partial(4)
    for (int j = 0; j < outputSize; j++) {
        activated_output[j] = std::exp(z_values[j] - maxValue);
        sum += activated_output[j];
    }

    #pragma omp unroll partial(4)
    for (int j = 0; j < outputSize; j++) {
        activated_output[j] /= sum;
    }
}

void DLINEAR(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* z_values = referenceDataBuffer.Data();
    float* output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        output[i] = 0.0f;
    }
}

void DRELU(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* input = referenceDataBuffer.Data();
    float* output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        output[i] = input[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void DSIGMOID(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* input = referenceDataBuffer.Data();
    float* output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        float s = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = s * (1.0f - s);
    }
}

void DTANH(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn) {
    float* input = referenceDataBuffer.Data();
    float* output = bufferToWriteOn.Data();
    int outputSize = bufferToWriteOn.ElementLength();

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        float t = std::tanh(input[i]);
        output[i] = 1.0f - (t * t);
    }
}

void DSOFTMAX(
    Napi::Float32Array& activatedOutput,
    Napi::Float32Array& incomingDelta,
    Napi::Float32Array& bufferToWriteOn
) {
    const float* probabilities = activatedOutput.Data();
    const float* delta = incomingDelta.Data();
    float* output = bufferToWriteOn.Data();
    int outputSize = activatedOutput.ElementLength();

    float dotProduct = 0.0f;

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        dotProduct += probabilities[i] * delta[i];
    }

    #pragma omp unroll partial(4)
    for (int i = 0; i < outputSize; i++) {
        output[i] = probabilities[i] * (delta[i] - dotProduct);
    }
}