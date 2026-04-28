

#include <napi.h>
#include <Eigen/Dense>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;
using Scalar = std::vector<int>;

using RowMatXf  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstMapM = Eigen::Map<const RowMatXf>;
using MapM      = Eigen::Map<RowMatXf>;

// NHWC im2col for a single sample.
// Output rows = output positions (oh,ow), cols = (kh, kw, c) flattened.
static void im2col_nhwc(const float* input, int H, int W, int C, int KH, int KW, int outH, int outW, int strides, float* patches) {
    const int patch_size = KH * KW * C;
    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            float* p = patches + (size_t)(oh * outW + ow) * patch_size;
            for (int kh = 0; kh < KH; ++kh) {
                int inY = oh * strides + kh;
                for (int kw = 0; kw < KW; ++kw) {
                    int inX = ow * strides + kw;
                    if ((unsigned)inY < (unsigned)H && (unsigned)inX < (unsigned)W) {
                        const float* src = input + (size_t)(inY * W + inX) * C;
                        std::memcpy(p, src, C * sizeof(float));
                    } else {
                        std::memset(p, 0, C * sizeof(float));
                    }
                    p += C;
                }
            }
        }
    }
}

Napi::Value ConvolveWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto input    = info[0].As<Napi::Float32Array>();
    auto kernels  = info[1].As<Napi::Float32Array>();
    auto biases   = info[2].As<Napi::Float32Array>();
    int strides = info[3].As<Napi::Number>().Int32Value();
    int outH    = info[4].As<Napi::Number>().Int32Value();
    int outW    = info[5].As<Napi::Number>().Int32Value();
    int F       = info[6].As<Napi::Number>().Int32Value();
    int KH      = info[7].As<Napi::Number>().Int32Value();
    int KW      = info[8].As<Napi::Number>().Int32Value();
    int C       = info[9].As<Napi::Number>().Int32Value();
    int H       = info[10].As<Napi::Number>().Int32Value();
    int W       = info[11].As<Napi::Number>().Int32Value();

    auto output = Napi::Float32Array::New(env, (size_t)outH * outW * F);

    const int patch_size = KH * KW * C;
    const int n_patches  = outH * outW;

    std::vector<float> patches((size_t)n_patches * patch_size);
    im2col_nhwc(input.Data(), H, W, C, KH, KW, outH, outW, strides, patches.data());

    ConstMapM Im(patches.data(),    n_patches, patch_size);
    ConstMapM K (kernels.Data(),    F,         patch_size);
    ConstMapM b (biases.Data(),     1,         F);
    MapM      Y (output.Data(),     n_patches, F);

    Y.noalias() = Im * K.transpose();
    Y.rowwise() += b.row(0);

    return output;
}

static Scalar ScalarArray(const Napi::Array& arr) {
    Scalar scalarArray;
    scalarArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        scalarArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return scalarArray;
}

Napi::Value ConvolveDeltaWrapper(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    auto padded_input    = info[0].As<Napi::Float32Array>();
    Scalar padded_shape  = ScalarArray(info[1].As<Napi::Array>());
    auto rotated_kernels = info[2].As<Napi::Float32Array>();
    Scalar kernel_shape  = ScalarArray(info[3].As<Napi::Array>());
    int oH = info[4].As<Napi::Number>().Int32Value();
    int oW = info[5].As<Napi::Number>().Int32Value();

    int Hp   = padded_shape[0];
    int Wp   = padded_shape[1];
    int C_in = padded_shape[2];

    int F   = kernel_shape[0];
    int KH  = kernel_shape[1];
    int KW  = kernel_shape[2];
    int C_k = kernel_shape[3];

    auto output = Napi::Float32Array::New(env, (size_t)oH * oW * C_k);

    const int patch_size = KH * KW * C_in;   // (kh, kw, f) flattened
    const int n_patches  = oH * oW;

    // 1) im2col the padded delta (stride 1, no real OOB but the helper handles it safely)
    std::vector<float> patches((size_t)n_patches * patch_size);
    im2col_nhwc(padded_input.Data(), Hp, Wp, C_in, KH, KW, oH, oW, /*strides=*/1, patches.data());

    // 2) Permute kernel from [F, KH, KW, C_k] -> [KH, KW, F, C_k]
    //    so its row order matches patches' column order.
    std::vector<float> K_perm((size_t)KH * KW * F * C_k);
    const float* Ksrc = rotated_kernels.Data();
    for (int f = 0; f < F; ++f) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                const float* src = Ksrc           + (((size_t)f*KH + kh)*KW + kw)*C_k;
                float*       dst = K_perm.data()  + (((size_t)kh*KW + kw)*F  + f)*C_k;
                std::memcpy(dst, src, C_k * sizeof(float));
            }
        }
    }

    // 3) GEMM: Y = Im * K_perm
    ConstMapM Im(patches.data(),  n_patches,  patch_size);
    ConstMapM K (K_perm.data(),   patch_size, C_k);
    MapM      Y (output.Data(),   n_patches,  C_k);

    Y.noalias() = Im * K;

    return output;
}


// int getInputIndex(int h, int w, int c, int W, int C) {
//     return (h * W + w) * C + c;
// }

// int getKernelIndex(int f, int kh, int kw, int c, int KH, int KW, int C) {
//     return ((f * KH + kh) * KW + kw) * C + c;
// }

// int getOutputIndex(int h, int w, int f, int W, int F) {
//     return (h * W + w) * F + f;
// }

// /* ==================== Wrappers ======================== */
// Napi::Value ConvolveWrapper(const Napi::CallbackInfo& info) {
//     Napi::Env env = info.Env();
//     Napi::Float32Array input = info[0].As<Napi::Float32Array>();
//     Napi::Float32Array kernels_array = info[1].As<Napi::Float32Array>();
//     Napi::Float32Array biases_array = info[2].As<Napi::Float32Array>();
//     int strides = info[3].As<Napi::Number>().Int32Value();
//     int output_height = info[4].As<Napi::Number>().Int32Value();
//     int output_width = info[5].As<Napi::Number>().Int32Value();
//     int num_filters = info[6].As<Napi::Number>().Int32Value();
//     int kernel_height = info[7].As<Napi::Number>().Int32Value();
//     int kernel_width = info[8].As<Napi::Number>().Int32Value();
//     int depth = info[9].As<Napi::Number>().Int32Value();
//     int input_height = info[10].As<Napi::Number>().Int32Value();
//     int input_width = info[11].As<Napi::Number>().Int32Value();

//     int expected_size = output_height * output_width * num_filters;

//     Napi::Float32Array output = Napi::Float32Array::New(env, expected_size);


//     float* data = input.Data();
//     float* kernels = kernels_array.Data();
//     float* biases = biases_array.Data();
//     float* outData = output.Data();

//     for (size_t f = 0; f < num_filters; f++) {
//         float bias = biases[f];

//         for (size_t oh = 0; oh < output_height; oh++) {
//             for (size_t ow = 0; ow < output_width; ow++) {
//                 float sum = 0.0f;

//                 for (size_t kh = 0; kh < kernel_height; kh++) {
//                     for (size_t kw = 0; kw < kernel_width; kw++) {
//                         for (size_t c = 0; c < depth; c++) {

//                             int inY = (oh * strides) + kh;
//                             int inX = (ow * strides) + kw;

//                             if (inY < input_height && inX < input_width) {
//                                 int input_idx = ((inY * input_width + inX) * depth + c);
//                                 int kernel_idx = (((f * kernel_height + kh) * kernel_width + kw ) * depth + c);

//                                 sum += data[input_idx] * kernels[kernel_idx];
//                             }
//                         }
//                     }
//                 }

//                 int outIndex = ((oh * output_width + ow) * num_filters + f);

//                 outData[outIndex] = sum + bias;
//             }
//         }
//     }

//     return output;

// }

// Napi::Value ConvolveDeltaWrapper(const Napi::CallbackInfo& info) {
//     Napi::Env env = info.Env();

//     Napi::Float32Array padded_input_array = info[0].As<Napi::Float32Array>();
//     Scalar padded_shape = ScalarArray(info[1].As<Napi::Array>());

//     Napi::Float32Array rotated_kernels_array = info[2].As<Napi::Float32Array>();
//     Scalar kernel_shape = ScalarArray(info[3].As<Napi::Array>());

//     size_t oH = info[4].As<Napi::Number>().Int32Value();
//     size_t oW = info[5].As<Napi::Number>().Int32Value();

//     size_t Hp = padded_shape[0];
//     size_t Wp = padded_shape[1];
//     size_t C_in = padded_shape[2];

//     size_t F = kernel_shape[0];
//     size_t KH = kernel_shape[1];
//     size_t KW = kernel_shape[2];
//     size_t C_k = kernel_shape[3];

//     size_t C = std::min(C_in, C_k);

//     // Infer output size (same as JS logic)
//     size_t H = Hp - KH + 1;
//     size_t W = Wp - KW + 1;

//     // Raw pointers (faster access)
//     float* padded = padded_input_array.Data();
//     float* rotatedKernels = rotated_kernels_array.Data();

//     // Create output array
//     Napi::Float32Array output = Napi::Float32Array::New(env, oH * oW * C_k);
//     float* out = output.Data();

//     // ---- Convolution ----
//     for (size_t c_out = 0; c_out < C_k; c_out++) {
//         for (size_t h = 0; h < oH; h++) {
//             for (size_t w = 0; w < oW; w++) {
//                 float sum = 0.0f;
//                 for (size_t kh = 0; kh < KH; kh++) {
//                     for (size_t kw = 0; kw < KW; kw++) {
//                         for (size_t f = 0; f < F; f++) {
//                             size_t ph = h + kh;
//                             size_t pw = w + kw;
//                             size_t inputIdx  = (ph * Wp + pw) * C_in + f;
//                             size_t kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
//                             sum += padded[inputIdx] * rotatedKernels[kernelIdx];
//                         }
//                     }
//                 }
//                 out[(h * oW + w) * C_k + c_out] = sum;
//             }
//         }
//     }
    
//     return output;
// }

/* ==================== Module exports ======================== */
void ConvolveRegister(Napi::Env env, Napi::Object exports) {
    exports.Set("Convolve", Napi::Function::New(env, ConvolveWrapper));
    exports.Set("ConvolveDelta", Napi::Function::New(env, ConvolveDeltaWrapper));
}