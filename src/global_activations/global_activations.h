#pragma once
#include <napi.h>


void LINEAR(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void RELU(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void SIGMOID(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void TANH(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void SOFTMAX(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void DLINEAR(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void DRELU(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void DSIGMOID(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void DTANH(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void DSOFTMAX(
    Napi::Float32Array& activatedOutput,
    Napi::Float32Array& incomingDelta,
    Napi::Float32Array& bufferToWriteOn
);
