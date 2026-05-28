__kernel void TransConv(
    __global const float* input,
    __global const float* kernels,
    __global const float* biases,
    __global float* output_tensor,
    const int strides,
    const int outputH,
    const int outputW,
    const int filters,
    const int kh,
    const int kw,
    const int depth,
    const int inputH,
    const int inputW
) {

    int oH = get_global_id(0);
    int oW = get_global_id(1);
    int f  = get_global_id(2);

    if (oH >= outputH || oW >= outputW || f >= filters) return;

    float sum = 0.0f;

    for (int ky = 0; ky < kh; ky++) {
        for (int kx = 0; kx < kw; kx++) {
            for (int c = 0; c < depth; c++) {

                int inY = oH * strides + ky;
                int inX = oW * strides + kx;

                if (inY < inputH && inX < inputW) {

                    int inputIndex = ((inY * inputW + inX) * depth + c);

                    int kernelIndex = (((f * kh + ky) * kw + kx) * depth + c);

                    sum += input[inputIndex] * kernels[kernelIndex];
                }
            }
        }
    }

    int outIndex = ((oH * outputW + oW) * filters + f);

    output_tensor[outIndex] = sum + biases[f];
}

__kernel void TransConvDelta(
    __global const float* delta,
    __global const float* kernels,
    __global float* output,
    const int outputHeight,
    const int outputWidth,
    const int Hp,
    const int Wp, 
    const int C_in,
    const int F,
    const int KH,
    const int KW,
    const int channel_kernel
) {

    int h = get_global_id(0);
    int w = get_global_id(1);
    int c_out = get_global_id(2);

    if (h >= outputHeight || w >= outputWidth || c_out >= channel_kernel)
        return;

    float sum = 0.0f;

    for (int kh = 0; kh < KH; kh++) {
        for (int kw = 0; kw < KW; kw++) {
            for (int f = 0; f < F; f++) {

                int ph = h + kh;
                int pw = w + kw;

                if (ph < Hp && pw < Wp) {

                    int padIdx = (ph * Wp + pw) * C_in + f;

                    int kernelIdx = ((f * KH + kh) * KW + kw) * channel_kernel + c_out;

                    sum += delta[padIdx] * kernels[kernelIdx];
                }
            }
        }
    }

    output[(h * outputWidth + w) * channel_kernel + c_out] = sum;
}