#include <napi.h>

void ActivationsRegister(Napi::Env env, Napi::Object exports);
void MatMulRegister(Napi::Env env, Napi::Object exports);
void ConvolveRegister(Napi::Env env, Napi::Object exports);
void GradientCalculationRegister(Napi::Env env, Napi::Object exports);
void OptimizerInternals(Napi::Env env, Napi::Object exports);
void Math_OPS(Napi::Env env, Napi::Object exports);
void Poolings(Napi::Env env, Napi::Object exports);
void detectGPU(Napi::Env env, Napi::Object exports);
void GpuLifecycleRegister(Napi::Env env, Napi::Object exports);
void utils(Napi::Env env, Napi::Object exports);
void _globals(Napi::Env env, Napi::Object exports);
void Embeddings(Napi::Env env, Napi::Object exports);
void loss(Napi::Env env, Napi::Object exports);
void recurrentFunc(Napi::Env env, Napi::Object exports);
void normalizers(Napi::Env env, Napi::Object exports);
void transConvFunc(Napi::Env env, Napi::Object exports);

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    ActivationsRegister(env, exports);
    MatMulRegister(env, exports);
    ConvolveRegister(env, exports);
    GradientCalculationRegister(env, exports);
    OptimizerInternals(env, exports);
    Math_OPS(env, exports);
    Poolings(env, exports);
    detectGPU(env, exports);
    utils(env, exports);
    GpuLifecycleRegister(env, exports);
    _globals(env, exports);
    Embeddings(env, exports);
    loss(env, exports);
    recurrentFunc(env, exports);
    normalizers(env, exports);
    transConvFunc(env, exports);
    return exports;
}

NODE_API_MODULE(neurex_core_native, Init);