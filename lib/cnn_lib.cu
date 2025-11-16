#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// CUDA KERNELS
// ============================================================================

// 2D Convolution kernel
__global__ void conv2d_kernel(
    const float* input,   // [batch, in_channels, height, width]
    const float* weights, // [out_channels, in_channels, kernel_h, kernel_w]
    const float* bias,    // [out_channels]
    float* output,        // [batch, out_channels, out_h, out_w]
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    int out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;

    if (idx < total) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c_out = (idx / (out_w * out_h)) % out_channels;
        int b = idx / (out_w * out_h * out_channels);

        float sum = bias[c_out];

        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                        int input_idx = ((b * in_channels + c_in) * input_h + h_in) * input_w + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }

        output[idx] = sum;
    }
}

// ReLU activation kernel (in-place)
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// 2D Max pooling kernel
__global__ void maxpool2d_kernel(
    const float* input,   // [batch, channels, height, width]
    float* output,        // [batch, channels, out_h, out_w]
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int pool_size,
    int stride
) {
    int out_h = (input_h - pool_size) / stride + 1;
    int out_w = (input_w - pool_size) / stride + 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;

    if (idx < total) {
        int w_out = idx % out_w;
        int h_out = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % channels;
        int b = idx / (out_w * out_h * channels);

        float max_val = -INFINITY;

        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int h_in = h_out * stride + ph;
                int w_in = w_out * stride + pw;
                int input_idx = ((b * channels + c) * input_h + h_in) * input_w + w_in;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }

        output[idx] = max_val;
    }
}

// Fully connected layer kernel
__global__ void fc_kernel(
    const float* input,   // [batch, in_features]
    const float* weights, // [out_features, in_features]
    const float* bias,    // [out_features]
    float* output,        // [batch, out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;

    if (idx < total) {
        int out_idx = idx % out_features;
        int b = idx / out_features;

        float sum = bias[out_idx];
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weights[out_idx * in_features + i];
        }

        output[idx] = sum;
    }
}

// Softmax kernel
__global__ void softmax_kernel(
    float* data,          // [batch, num_classes]
    int batch_size,
    int num_classes
) {
    int b = blockIdx.x;

    if (b < batch_size) {
        // Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, data[b * num_classes + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            data[b * num_classes + i] = expf(data[b * num_classes + i] - max_val);
            sum += data[b * num_classes + i];
        }

        // Normalize
        for (int i = 0; i < num_classes; i++) {
            data[b * num_classes + i] /= sum;
        }
    }
}

// ============================================================================
// C WRAPPER FUNCTIONS (exposed to Python via ctypes)
// ============================================================================

extern "C" {

void cuda_conv2d(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    int out_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    int out_w = (input_w + 2 * padding - kernel_w) / stride + 1;
    int total = batch_size * out_channels * out_h * out_w;

    float *d_input, *d_weights, *d_bias, *d_output;

    cudaMalloc(&d_input, batch_size * in_channels * input_h * input_w * sizeof(float));
    cudaMalloc(&d_weights, out_channels * in_channels * kernel_h * kernel_w * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * in_channels * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, out_channels * in_channels * kernel_h * kernel_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        stride, padding
    );

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

void cuda_relu(float* data, int size) {
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_data, size);

    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void cuda_maxpool2d(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int pool_size,
    int stride
) {
    int out_h = (input_h - pool_size) / stride + 1;
    int out_w = (input_w - pool_size) / stride + 1;
    int total = batch_size * channels * out_h * out_w;

    float *d_input, *d_output;
    cudaMalloc(&d_input, batch_size * channels * input_h * input_w * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * channels * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool2d_kernel<<<blocks, threads>>>(
        d_input, d_output,
        batch_size, channels,
        input_h, input_w,
        pool_size, stride
    );

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

void cuda_fc(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int total = batch_size * out_features;

    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * in_features * sizeof(float));
    cudaMalloc(&d_weights, out_features * in_features * sizeof(float));
    cudaMalloc(&d_bias, out_features * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_features * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fc_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_features, out_features
    );

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
}

void cuda_softmax(float* data, int batch_size, int num_classes) {
    float* d_data;
    cudaMalloc(&d_data, batch_size * num_classes * sizeof(float));
    cudaMemcpy(d_data, data, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<batch_size, 1>>>(d_data, batch_size, num_classes);

    cudaDeviceSynchronize();
    cudaMemcpy(data, d_data, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// Cross-entropy loss kernel
void cuda_cross_entropy_loss(
    const float* predictions,  // [batch, num_classes]
    const int* labels,         // [batch]
    float* loss,               // [1] output
    int batch_size,
    int num_classes
) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int label = labels[b];
        float pred = predictions[b * num_classes + label];
        total_loss += -logf(fmaxf(pred, 1e-10f));  // Avoid log(0)
    }
    *loss = total_loss / batch_size;
}

// SGD parameter update kernel
__global__ void sgd_update_kernel(
    float* params,
    const float* gradients,
    float learning_rate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * gradients[idx];
    }
}

void cuda_sgd_update(
    float* params,
    const float* gradients,
    float learning_rate,
    int size
) {
    float *d_params, *d_gradients;
    cudaMalloc(&d_params, size * sizeof(float));
    cudaMalloc(&d_gradients, size * sizeof(float));

    cudaMemcpy(d_params, params, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, gradients, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(d_params, d_gradients, learning_rate, size);

    cudaDeviceSynchronize();
    cudaMemcpy(params, d_params, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_params);
    cudaFree(d_gradients);
}

} // extern "C"
