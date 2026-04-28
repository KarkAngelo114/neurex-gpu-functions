#include <napi.h>
#include <cmath>
using namespace std;
using Scalar = std::vector<int>;

static Scalar ScalarArray(const Napi::Array& arr) {
    Scalar scalarArray;
    scalarArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        scalarArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return scalarArray;
}

Napi::Value MaxPoolingWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    Scalar pool_size = ScalarArray(info[1].As<Napi::Array>());
    Scalar inputShape = ScalarArray(info[2].As<Napi::Array>());
    Scalar outputShape = ScalarArray(info[3].As<Napi::Array>());
    size_t strides = info[4].As<Napi::Number>().Int32Value();
    
    size_t poolH = pool_size[0];
    size_t poolW = pool_size[1];

    size_t inputH = inputShape[0];
    size_t inputW = inputShape[1];
    size_t inputD = inputShape[2];

    size_t outputH = outputShape[0];
    size_t outputW = outputShape[1];
    size_t outputD = outputShape[2];

    // prepare output
    Napi::Float32Array output = Napi::Float32Array::New(env, outputH * outputW * outputD);
    Napi::Float32Array maxArray = Napi::Float32Array::New(env, outputH * outputW * outputD);

    float* arr = input_array.Data();
    float* max = maxArray.Data();
    float* out = output.Data();

    for (size_t d = 0; d < inputD; d++) {
        for (size_t i = 0; i < outputH; i++) {
            for (size_t j = 0;  j < outputW; j++) {
                float maxVal = -INFINITY;
                int maxIdx = -1;

                size_t startH = i * strides;
                size_t startW = j * strides;

                for (size_t ph = 0; ph < poolH; ph++) {
                    for (size_t pw = 0; pw < poolW; pw++) {
                        int currH = startH + ph;
                        int currW = startW + pw;

                        // Check bounds to handle cases where window might exceed input dimensions
                        if (currH < inputH && currW < inputW) {
                            // Calculate index in the flattened 1D array
                            int idx = (currH * inputW * inputD) + (currW * inputD) + d;
                            float val = arr[idx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }
                }
                int outIdx = (i * outputW * outputD) + (j * outputD) + d;
                out[outIdx] = maxVal == -INFINITY ? 0 : maxVal;
                max[outIdx] = maxIdx;
                
            }
        }
    }

    Napi::Object objectOutput= Napi::Object::New(env);
    objectOutput.Set("output", output);
    objectOutput.Set("maxIndices", maxArray);

    return objectOutput;
}



void Poolings(Napi::Env env, Napi::Object exports) {
    exports.Set("MaxPooling", Napi::Function::New(env, MaxPoolingWrapper));
}