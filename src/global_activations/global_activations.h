#pragma once
#include <napi.h>


void LINEAR(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void RELU(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void SIGMOID(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void TANH(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
void SOFTMAX(Napi::Float32Array& referenceDataBuffer, Napi::Float32Array& bufferToWriteOn);
