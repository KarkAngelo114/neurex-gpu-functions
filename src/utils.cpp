#include <napi.h>
#include <vector>
using Array = std::vector<int>;

static Array Vectorize(const Napi::Array& arr) {
    Array vectorArray;
    vectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        vectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return vectorArray;
}

Napi::Value DilateInputWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    Array shape = Vectorize(info[1].As<Napi::Array>());
    size_t stride = info[2].As<Napi::Number>().Int32Value();

    size_t H = shape[0];
    size_t W = shape[1];
    size_t C = shape[2];

    size_t dilatedH = H * stride + (H - 1) * (stride - 1);
    size_t dilatedW = W * stride + (W - 1) * (stride - 1);
    
    int dilatedSize = dilatedH * dilatedW * C;

    Napi::Float32Array dilatedOutput = Napi::Float32Array::New(env, dilatedSize);

    float* input = input_arr.Data();
    float* dilated = dilatedOutput.Data();

    for (size_t c = 0; c < C; c++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t w = 0; w < W; w++) {
                size_t srcIdx = (h * W + w) * C + c;
                size_t dilatedHIdx = h * stride;
                size_t dilatedWIdx = w * stride;
                size_t dstIdx = (dilatedHIdx * dilatedW + dilatedWIdx) * C + c;
                dilated[dstIdx] = input[srcIdx] || 0;
            }
        }
    }

    return dilatedOutput;
}

Napi::Value ApplyPadding(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_arr = info[0].As<Napi::Float32Array>();
    size_t inputH = info[1].As<Napi::Number>().Int32Value();
    size_t inputW = info[2].As<Napi::Number>().Int32Value();
    size_t channels = info[3].As<Napi::Number>().Int32Value();
    size_t padTop = info[4].As<Napi::Number>().Int32Value();
    size_t padBottom = info[5].As<Napi::Number>().Int32Value();
    size_t padLeft = info[6].As<Napi::Number>().Int32Value();
    size_t padRight = info[7].As<Napi::Number>().Int32Value();

    size_t newH = inputH + padTop + padBottom;
    size_t newW = inputW + padLeft + padRight;
    Napi::Float32Array outputData = Napi::Float32Array::New(env, newH * newW * channels);

    float* input = input_arr.Data();
    float* output = outputData.Data();

    for (size_t i = 0; i < inputH; i++) {
        for (size_t j = 0; j < inputW; j++) {
            for (size_t c = 0; c < channels; c++) {
                size_t oldIdx = (i * inputW + j) * channels + c;
                size_t newIdx = ((i + padTop) * newW + (j + padLeft)) * channels + c;
                output[newIdx] = input[oldIdx];
            }
        }
    }

    // Create the vector as you intended (using curly braces for initialization)
    Array newShape = { static_cast<int>(newH), static_cast<int>(newW), static_cast<int>(channels) };

    Napi::Object data = Napi::Object::New(env);
    data.Set("data", outputData);

    // Convert std::vector<int> to Napi::Array
    Napi::Array shapeArray = Napi::Array::New(env, newShape.size());
    for (size_t i = 0; i < newShape.size(); i++) {
        shapeArray.Set(i, Napi::Number::New(env, newShape[i]));
    }

    data.Set("shape", shapeArray);

    return data;

}


void utils(Napi::Env env, Napi::Object exports) {
    exports.Set("DilateDelta", Napi::Function::New(env, DilateInputWrapper));
    exports.Set("ApplyPadding", Napi::Function::New(env, ApplyPadding));
}