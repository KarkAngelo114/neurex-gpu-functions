#include <napi.h>
#include <vector>
#include "../gpu/gpu_context.h"
#include "globals.h"
using Array = std::vector<float>;
using Matrix = std::vector<Array>;

static Matrix global_output_Tensor;
static bool global_boolean_On_GPU_state;

// get the corresponding output tensor template using a pointer
const Array& getGlobalOutputTensors(int pointer) {
    return global_output_Tensor[pointer];
}
// get the boolean state initiated by JS
bool get_Global_Boolean_On_GPU() {
    return global_boolean_On_GPU_state;
}


Napi::Value setOutputTemplateTensors(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Array outputTensors = info[1].As<Napi::Array>();
    uint32_t globalOutputTensorSize = outputTensors.Length();
    global_output_Tensor.resize(globalOutputTensorSize);

    for (uint32_t i = 0; i < globalOutputTensorSize; i++) {
        Napi::Float32Array output_tensor_templates = outputTensors.Get(i).As<Napi::Float32Array>();

        global_output_Tensor[i].assign(output_tensor_templates.Data(), output_tensor_templates.Data() + output_tensor_templates.ElementLength());
    }

    if (get_Global_Boolean_On_GPU()) {
        std::string err;
        bool ok = GpuContext::instance().uploadOutputTemplates(global_output_Tensor, err);
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
    exports.Set("uploadOutputTensorTemplates", Napi::Function::New(env, setOutputTemplateTensors));
    exports.Set("setOnGPU", Napi::Function::New(env, setOnGPU_Boolean_State));
}