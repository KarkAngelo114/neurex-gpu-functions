#include <napi.h>
#include "gpu/gpu_context.h"
#include "globals/globals.h"
#include <vector>
using IntArray = std::vector<int>;
using FloatArray = std::vector<float>;

static IntArray Vectorize(const Napi::Array& arr) {
    IntArray VectorArray;
    VectorArray.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
        VectorArray.push_back(arr.Get(i).As<Napi::Number>().Int32Value());
    }
    return VectorArray;
}

static FloatArray Rotate_kernels(int F, int KH, int KW, int D, int pointer) {
    const auto& kernels_arr = getGlobalWeights(pointer);

    size_t kernel_length = kernels_arr.size();

    FloatArray outputData(kernel_length);

    const float* kernels = kernels_arr.data();
    float* rotated = outputData.data();

    for (size_t f = 0; f < F; f++) {
        for (size_t kh = 0; kh < KH; kh++) {
            for (size_t kw = 0; kw < KW; kw++) {
                for (size_t d = 0; d < D; d++) {
                    // Original Index
                    size_t oldIdx = (f * KH * KW * D) + (kh * KW * D) + (kw * D) + d;
                    
                    // Rotated Index (Flip KH and KW)
                    size_t newKh = KH - 1 - kh;
                    size_t newKw = KW - 1 - kw;
                    size_t newIdx = (f * KH * KW * D) + (newKh * KW * D) + (newKw * D) + d;
                    
                    rotated[newIdx] = kernels[oldIdx];
                }
            }
        }
    }

    return outputData;
}

Napi::Value TransConv_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    int outputH = info[2].As<Napi::Number>().Int32Value();
    int outputW = info[3].As<Napi::Number>().Int32Value();
    int num_filters = info[4].As<Napi::Number>().Int32Value();
    int kernel_height = info[5].As<Napi::Number>().Int32Value();
    int kernel_width = info[6].As<Napi::Number>().Int32Value();
    int depth = info[7].As<Napi::Number>().Int32Value();
    int inputH = info[8].As<Napi::Number>().Int32Value();
    int inputW = info[9].As<Napi::Number>().Int32Value();
    int pointer = info[10].As<Napi::Number>().Int32Value();
    int output_template_pointer = info[11].As<Napi::Number>().Int32Value();
    FloatArray kernel_array = Rotate_kernels(num_filters, kernel_height, kernel_width, depth, pointer);
    int totalSize = outputH * outputW * num_filters;
    Napi::Float32Array output = Napi::Float32Array::New(env, totalSize);

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("TransConv");
    

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_array.ElementLength(), input_array.Data(), nullptr);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* kernel_array.size(), kernel_array.data(), nullptr);
    cl_mem biases =  gpu.bias(pointer);
    cl_mem output_tensor = gpu.output(output_template_pointer);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biases);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_tensor);
    clSetKernelArg(kernel, 4, sizeof(int), &outputH);
    clSetKernelArg(kernel, 5, sizeof(int), &outputW);
    clSetKernelArg(kernel, 6, sizeof(int), &num_filters);
    clSetKernelArg(kernel, 7, sizeof(int), &kernel_height);
    clSetKernelArg(kernel, 8, sizeof(int), &kernel_width);
    clSetKernelArg(kernel, 9, sizeof(int), &depth);
    clSetKernelArg(kernel, 10, sizeof(int), &inputH);
    clSetKernelArg(kernel, 11, sizeof(int), &inputW);

    size_t globalSize[3] = {
        (size_t)outputH,
        (size_t)outputW,
        (size_t)num_filters
    };
    clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output_tensor, CL_TRUE, 0, sizeof(float)* totalSize, output.Data(), 0, nullptr, nullptr);
    clFinish(queue);
    clReleaseMemObject(input);
    clReleaseMemObject(weights);

    return output;
}

Napi::Value TransConv_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    int strides = info[1].As<Napi::Number>().Int32Value();
    int outputH = info[2].As<Napi::Number>().Int32Value();
    int outputW = info[3].As<Napi::Number>().Int32Value();
    int num_filters = info[4].As<Napi::Number>().Int32Value();
    int kernel_height = info[5].As<Napi::Number>().Int32Value();
    int kernel_width = info[6].As<Napi::Number>().Int32Value();
    int depth = info[7].As<Napi::Number>().Int32Value();
    int inputH = info[8].As<Napi::Number>().Int32Value();
    int inputW = info[9].As<Napi::Number>().Int32Value();
    int pointer = info[10].As<Napi::Number>().Int32Value();
    int output_template_pointer = info[11].As<Napi::Number>().Int32Value();

    // get the kernels from the global store and rotate
    FloatArray rotatedKernels = Rotate_kernels(num_filters, kernel_height, kernel_width, depth, pointer);

    // get biases from the global store
    FloatArray biases_array = getGlobalBiases(pointer);

    // get the output tensor template from the global store and cast to Napi::Float32Array
    FloatArray output_template = getGlobalOutputTensors(output_template_pointer);

    Napi::Float32Array output = Napi::Float32Array::New(env, output_template.size());
    std::memcpy(
        output.Data(),
        output_template.data(),
        output_template.size() * sizeof(float)
    );

    float* input = input_array.Data();
    float* kernel = rotatedKernels.data();
    float* biases = biases_array.data();
    float* outputData = output.Data();

    for (int f = 0; f < num_filters; f++) {

        float bias = biases[f];

        for (int y = 0; y < outputH; y++) {
            for (int x = 0; x < outputW; x++) {

                float sum = 0.0f;

                for (int ky = 0; ky < kernel_height; ky++) {
                    for (int kx = 0; kx < kernel_width; kx++) {
                        for (int c = 0; c < depth; c++) {

                            int inY = y + ky;
                            int inX = x + kx;

                            if (inY < inputH && inX < inputW) {

                                int inputIndex = ((inY * inputW + inX) * depth + c);

                                int kernelIndex = (((f * kernel_height + ky)* kernel_width + kx)* depth + c);

                                sum += input[inputIndex] * kernel[kernelIndex];
                            }
                        }
                    }
                }

                int outIndex = ((y * outputW + x) * num_filters + f);

                outputData[outIndex] = sum + bias;
            }
        }
    }

    return output;
}

Napi::Value TransConvDelta_GPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    IntArray shape = Vectorize(info[1].As<Napi::Array>()); 
    IntArray kernels_shape = Vectorize(info[2].As<Napi::Array>()); 
    int oH = info[3].As<Napi::Number>().Int32Value();
    int oW = info[4].As<Napi::Number>().Int32Value(); 
    int pointer = info[5].As<Napi::Number>().Int32Value();
    
    int Hp = shape[0];
    int Wp = shape[1];
    int C_in = shape[2];

    int F = kernels_shape[0];
    int KH = kernels_shape[1];
    int KW = kernels_shape[2];
    int C_k = kernels_shape[3];
    
    FloatArray kernels = Rotate_kernels(F, KH, KW, C_k, pointer);
    Napi::Float32Array output = Napi::Float32Array::New(env, oH * oW * C_k);

    auto& gpu = GpuContext::instance();
    cl_context context = gpu.context();
    cl_command_queue queue = gpu.queue();
    cl_kernel kernel = gpu.kernel("TransConvDelta");

    cl_mem delta = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* input_array.ElementLength(), input_array.Data(), nullptr);
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)* kernels.size(), kernels.data(), nullptr);
    cl_mem output_tensor = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.ElementLength(), nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &delta);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_tensor);
    clSetKernelArg(kernel, 3, sizeof(int), &oH);
    clSetKernelArg(kernel, 4, sizeof(int), &oW);
    clSetKernelArg(kernel, 5, sizeof(int), &Hp);
    clSetKernelArg(kernel, 6, sizeof(int), &Wp);
    clSetKernelArg(kernel, 7, sizeof(int), &C_in);
    clSetKernelArg(kernel, 8, sizeof(int), &F);
    clSetKernelArg(kernel, 9, sizeof(int), &KH);
    clSetKernelArg(kernel, 10, sizeof(int), &KW);
    clSetKernelArg(kernel, 11, sizeof(int), &C_k);

    size_t globalSize[3] = {
        (size_t)oH,
        (size_t)oW,
        (size_t)C_k
    };
    clEnqueueNDRangeKernel(queue, kernel, 3, 0, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, output_tensor, CL_TRUE, 0, sizeof(float)* output.ElementLength(), output.Data(), 0, nullptr, nullptr);

    clFinish(queue);
    clReleaseMemObject(delta);
    clReleaseMemObject(weights);
    clReleaseMemObject(output_tensor);

    return output;
}

Napi::Value TransConvDelta_CPU(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Float32Array input_array = info[0].As<Napi::Float32Array>();
    IntArray shape = Vectorize(info[1].As<Napi::Array>()); 
    IntArray kernels_shape = Vectorize(info[2].As<Napi::Array>()); 
    int oH = info[3].As<Napi::Number>().Int32Value();
    int oW = info[4].As<Napi::Number>().Int32Value(); 
    int pointer = info[5].As<Napi::Number>().Int32Value();
    
    int Hp = shape[0];
    int Wp = shape[1];
    int C_in = shape[2];

    int F = kernels_shape[0];
    int KH = kernels_shape[1];
    int KW = kernels_shape[2];
    int C_k = kernels_shape[3];
    
    FloatArray kernels = Rotate_kernels(F, KH, KW, C_k, pointer);
    Napi::Float32Array output_tensor = Napi::Float32Array::New(env, oH * oW * C_k);

    float* input = input_array.Data();
    float* rotated_kernel = kernels.data();
    float* output = output_tensor.Data();

    for (int c_out = 0; c_out < C_k; c_out++) {
        for (int h = 0; h < oH; h++) {
            for (int w = 0; w < oW; w++) {
                float sum = 0.0f;
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        for (int f = 0; f < F; f++) {
                            int ph = h + kh, pw = w + kw;
                            if (ph < Hp && pw < Wp) { 
                                int padIdx = (ph * Wp + pw) * C_in + f;
                                int kernelIdx = ((f * KH + kh) * KW + kw) * C_k + c_out;
                                sum += input[padIdx] * rotated_kernel[kernelIdx];
                            }
                        }
                    }
                }
                output[(h * oW + w) * C_k + c_out] = sum;
            }
        }
    }

    return output_tensor;   
}

Napi::Value TransConvWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return TransConv_GPU(info);
    }

    return TransConv_CPU(info);
}

Napi::Value TransConvDeltaWrapper(const Napi::CallbackInfo& info) {
    if (get_Global_Boolean_On_GPU()) {
        return TransConvDelta_GPU(info);
    }

    return TransConvDelta_CPU(info);
}

void TransConvFunc(const Napi::Env env, const Napi::Object exports) {
    exports.Set("TransConv", Napi::Function::New(env, TransConvWrapper));
    exports.Set("TransConvolveDelta", Napi::Function::New(env, TransConvDeltaWrapper));
}