#include <napi.h>
#include <omp.h>
#include <vector>

Napi::Value element_wise_mul_wrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    Napi::Float32Array arr1 = info[0].As<Napi::Float32Array>();
    Napi::Float32Array arr2 = info[1].As<Napi::Float32Array>();
    int arr_length = arr1.ElementLength();
    Napi::Float32Array output = Napi::Float32Array::New(env, arr_length);

    float* a1 = arr1.Data();
    float* a2 = arr2.Data();
    float* o = output.Data();

    #pragma omp for schedule(static)
    for (int i = 0; i < arr_length; i++) {
        o[i] = a1[i] * a2[i];
    }

    return output;

}

void Math_OPS(Napi::Env env, Napi::Object exports) {
    exports.Set("element_wise_mul", Napi::Function::New(env, element_wise_mul_wrapper));
}
