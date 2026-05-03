#include <napi.h>
#include <vector>
#include "../gpu/gpu_context.h"
#include "globals.h"
using Array = std::vector<float>;
using Matrix = std::vector<Array>;

static Matrix global_Weights;
static Matrix global_biases;
static Matrix global_output_Tensor;
static bool global_boolean_On_GPU_state;

// get the corresponding weight params using a pointer
const Array& getGlobalWeights(int pointer) {
    return global_Weights[pointer];
}

// get the corresponding biase params using a pointer
const Array& getGlobalBiases(int pointer) {
    return global_biases[pointer];
}

// get the corresponding output tensor template using a pointer
const Array& getGlobalOutputTensors(int pointer) {
    return global_output_Tensor[pointer];
}
// get the boolean state initiated by JS
bool get_Global_Boolean_On_GPU() {
    return global_boolean_On_GPU_state;
}




Napi::Value setGlobalParams(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Array weights = info[0].As<Napi::Array>();
    Napi::Array biases = info[1].As<Napi::Array>();
    Napi::Array outputTensors = info[2].As<Napi::Array>();

    uint32_t globalParamSize = weights.Length();
    uint32_t globalOutputTensorSize = outputTensors.Length();

    global_Weights.resize(globalParamSize);
    global_biases.resize(globalParamSize);
    global_output_Tensor.resize(globalOutputTensorSize);

    for (uint32_t i = 0; i < globalParamSize; i++) {
        Napi::Float32Array w = weights.Get(i).As<Napi::Float32Array>();

        global_Weights[i].assign(w.Data(), w.Data() + w.ElementLength());

        Napi::Float32Array b = biases.Get(i).As<Napi::Float32Array>();

        global_biases[i].assign(b.Data(), b.Data() + b.ElementLength());
    }

    for (uint32_t i = 0; i < globalOutputTensorSize; i++) {
        Napi::Float32Array output_tensor_templates = outputTensors.Get(i).As<Napi::Float32Array>();

        global_output_Tensor[i].assign(output_tensor_templates.Data(), output_tensor_templates.Data() + output_tensor_templates.ElementLength());
    }

    if (get_Global_Boolean_On_GPU()) {
        std::string err;
        bool ok = GpuContext::instance().uploadParams(global_Weights, global_biases, global_output_Tensor, err);
        if (!ok) {
            Napi::TypeError::New(env, err).ThrowAsJavaScriptException();
        }
    }
    

    return env.Undefined();
}



Napi::Value setOnGPU_Boolean_State(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    global_boolean_On_GPU_state = info[0].As<Napi::Boolean>().Value();

    return env.Undefined();
}

void _globals(Napi::Env env, Napi::Object exports) {
    exports.Set("setGlobalParams", Napi::Function::New(env, setGlobalParams));
    exports.Set("setOnGPU", Napi::Function::New(env, setOnGPU_Boolean_State));
}