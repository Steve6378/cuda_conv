#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================================
// ERROR CHECKING MACRO
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Launch Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Execution Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

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

    CUDA_CHECK(cudaMalloc(&d_input, batch_size * in_channels * input_h * input_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, out_channels * in_channels * kernel_h * kernel_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, batch_size * in_channels * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, out_channels * in_channels * kernel_h * kernel_w * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w,
        stride, padding
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

void cuda_relu(float* data, int size) {
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_data, size);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
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
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * channels * input_h * input_w * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, batch_size * channels * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool2d_kernel<<<blocks, threads>>>(
        d_input, d_output,
        batch_size, channels,
        input_h, input_w,
        pool_size, stride
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
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
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias, out_features * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fc_kernel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        batch_size, in_features, out_features
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(output, d_output, total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

void cuda_softmax(float* data, int batch_size, int num_classes) {
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, batch_size * num_classes * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, data, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice));

    softmax_kernel<<<batch_size, 1>>>(d_data, batch_size, num_classes);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(data, d_data, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

// Softmax + Cross-entropy backward (combined for numerical stability)
__global__ void softmax_cross_entropy_backward_kernel(
    const float* predictions,  // [batch, num_classes] - softmax output
    const int* labels,         // [batch]
    float* grad_output,        // [batch, num_classes] - gradient output
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;

    if (idx < total) {
        int class_idx = idx % num_classes;
        int b = idx / num_classes;
        int true_label = labels[b];

        // d(loss)/d(logits) for softmax + cross-entropy
        float grad = predictions[idx];
        if (class_idx == true_label) {
            grad -= 1.0f;
        }
        grad_output[idx] = grad / batch_size;  // Average over batch
    }
}

void cuda_softmax_cross_entropy_backward(
    const float* predictions,
    const int* labels,
    float* grad_output,
    int batch_size,
    int num_classes
) {
    int total = batch_size * num_classes;

    float *d_predictions, *d_grad_output;
    int *d_labels;

    CUDA_CHECK(cudaMalloc(&d_predictions, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_predictions, predictions, total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels, batch_size * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_cross_entropy_backward_kernel<<<blocks, threads>>>(
        d_predictions, d_labels, d_grad_output, batch_size, num_classes
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(grad_output, d_grad_output, total * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_grad_output));
}

// Cross-entropy loss (forward)
void cuda_cross_entropy_loss(
    const float* predictions,
    const int* labels,
    float* loss,
    int batch_size,
    int num_classes
) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int label = labels[b];
        float pred = predictions[b * num_classes + label];
        total_loss += -logf(fmaxf(pred, 1e-10f));
    }
    *loss = total_loss / batch_size;
}

// FC backward
__global__ void fc_backward_kernel(
    const float* grad_output,  // [batch, out_features]
    const float* input,        // [batch, in_features]
    const float* weights,      // [out_features, in_features]
    float* grad_input,         // [batch, in_features]
    float* grad_weights,       // [out_features, in_features]
    float* grad_bias,          // [out_features]
    int batch_size,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute grad_input
    int total_input = batch_size * in_features;
    if (idx < total_input) {
        int in_idx = idx % in_features;
        int b = idx / in_features;

        float grad = 0.0f;
        for (int o = 0; o < out_features; o++) {
            grad += grad_output[b * out_features + o] * weights[o * in_features + in_idx];
        }
        grad_input[idx] = grad;
    }

    // Compute grad_weights (use atomic adds for parallel safety)
    int total_weights = out_features * in_features;
    if (idx < total_weights) {
        int in_idx = idx % in_features;
        int out_idx = idx / in_features;

        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += grad_output[b * out_features + out_idx] * input[b * in_features + in_idx];
        }
        grad_weights[idx] = grad;
    }

    // Compute grad_bias
    if (idx < out_features) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += grad_output[b * out_features + idx];
        }
        grad_bias[idx] = grad;
    }
}

void cuda_fc_backward(
    const float* grad_output,
    const float* input,
    const float* weights,
    float* grad_input,
    float* grad_weights,
    float* grad_bias,
    int batch_size,
    int in_features,
    int out_features
) {
    int max_size = fmaxf(batch_size * in_features, out_features * in_features);
    max_size = fmaxf(max_size, out_features);

    float *d_grad_output, *d_input, *d_weights, *d_grad_input, *d_grad_weights, *d_grad_bias;

    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, out_features * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output, batch_size * out_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, batch_size * in_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, out_features * in_features * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (max_size + threads - 1) / threads;

    fc_backward_kernel<<<blocks, threads>>>(
        d_grad_output, d_input, d_weights,
        d_grad_input, d_grad_weights, d_grad_bias,
        batch_size, in_features, out_features
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(grad_input, d_grad_input, batch_size * in_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_weights, d_grad_weights, out_features * in_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_bias, d_grad_bias, out_features * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_bias));
}

// ReLU backward
__global__ void relu_backward_kernel(
    const float* grad_output,  // gradient from next layer
    const float* input,        // input to ReLU (before activation)
    float* grad_input,         // gradient to pass back
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void cuda_relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    float *d_grad_output, *d_input, *d_grad_input;

    CUDA_CHECK(cudaMalloc(&d_grad_output, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(d_grad_output, d_input, d_grad_input, size);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(grad_input, d_grad_input, size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// MaxPool backward
__global__ void maxpool2d_backward_kernel(
    const float* grad_output,  // [batch, channels, out_h, out_w]
    const float* input,        // [batch, channels, input_h, input_w]
    const float* output,       // [batch, channels, out_h, out_w] - forward output
    float* grad_input,         // [batch, channels, input_h, input_w]
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

        float max_val = output[idx];
        float grad = grad_output[idx];

        // Find which input position produced the max
        bool found = false;
        for (int ph = 0; ph < pool_size && !found; ph++) {
            for (int pw = 0; pw < pool_size && !found; pw++) {
                int h_in = h_out * stride + ph;
                int w_in = w_out * stride + pw;
                int input_idx = ((b * channels + c) * input_h + h_in) * input_w + w_in;

                if (input[input_idx] == max_val) {
                    atomicAdd(&grad_input[input_idx], grad);
                    found = true;  // Only one position gets the gradient
                }
            }
        }
    }
}

void cuda_maxpool2d_backward(
    const float* grad_output,
    const float* input,
    const float* output,
    float* grad_input,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int pool_size,
    int stride
) {
    int out_h = (input_h - pool_size) / stride + 1;
    int out_w = (input_w - pool_size) / stride + 1;
    int total_out = batch_size * channels * out_h * out_w;
    int total_in = batch_size * channels * input_h * input_w;

    float *d_grad_output, *d_input, *d_output, *d_grad_input;

    CUDA_CHECK(cudaMalloc(&d_grad_output, total_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, total_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, total_in * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output, total_out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, total_in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, output, total_out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_input, 0, total_in * sizeof(float)));  // Initialize to zero

    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads>>>(
        d_grad_output, d_input, d_output, d_grad_input,
        batch_size, channels, input_h, input_w, pool_size, stride
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(grad_input, d_grad_input, total_in * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grad_input));
}

// Conv2D backward - compute grad_input
__global__ void conv2d_backward_input_kernel(
    const float* grad_output,  // [batch, out_channels, out_h, out_w]
    const float* weights,      // [out_channels, in_channels, kernel_h, kernel_w]
    float* grad_input,         // [batch, in_channels, input_h, input_w]
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
    int total = batch_size * in_channels * input_h * input_w;

    if (idx < total) {
        int w_in = idx % input_w;
        int h_in = (idx / input_w) % input_h;
        int c_in = (idx / (input_w * input_h)) % in_channels;
        int b = idx / (input_w * input_h * in_channels);

        float grad = 0.0f;

        // For each output channel and position that uses this input position
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Which output positions are affected by this input position?
                    int h_out = (h_in + padding - kh);
                    int w_out = (w_in + padding - kw);

                    if (h_out % stride == 0 && w_out % stride == 0) {
                        h_out /= stride;
                        w_out /= stride;

                        if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w) {
                            int grad_out_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                            int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                            grad += grad_output[grad_out_idx] * weights[weight_idx];
                        }
                    }
                }
            }
        }

        grad_input[idx] = grad;
    }
}

// Conv2D backward - compute grad_weights
__global__ void conv2d_backward_weights_kernel(
    const float* grad_output,  // [batch, out_channels, out_h, out_w]
    const float* input,        // [batch, in_channels, input_h, input_w]
    float* grad_weights,       // [out_channels, in_channels, kernel_h, kernel_w]
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
    int total = out_channels * in_channels * kernel_h * kernel_w;

    if (idx < total) {
        int kw = idx % kernel_w;
        int kh = (idx / kernel_w) % kernel_h;
        int c_in = (idx / (kernel_w * kernel_h)) % in_channels;
        int c_out = idx / (kernel_w * kernel_h * in_channels);

        float grad = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int h_in = oh * stride - padding + kh;
                    int w_in = ow * stride - padding + kw;

                    if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                        int grad_out_idx = ((b * out_channels + c_out) * out_h + oh) * out_w + ow;
                        int input_idx = ((b * in_channels + c_in) * input_h + h_in) * input_w + w_in;
                        grad += grad_output[grad_out_idx] * input[input_idx];
                    }
                }
            }
        }

        grad_weights[idx] = grad;
    }
}

// Conv2D backward - compute grad_bias
__global__ void conv2d_backward_bias_kernel(
    const float* grad_output,  // [batch, out_channels, out_h, out_w]
    float* grad_bias,          // [out_channels]
    int batch_size,
    int out_channels,
    int out_h,
    int out_w
) {
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (c_out < out_channels) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int out_idx = ((b * out_channels + c_out) * out_h + h) * out_w + w;
                    grad += grad_output[out_idx];
                }
            }
        }
        grad_bias[c_out] = grad;
    }
}

void cuda_conv2d_backward(
    const float* grad_output,
    const float* input,
    const float* weights,
    float* grad_input,
    float* grad_weights,
    float* grad_bias,
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

    int input_size = batch_size * in_channels * input_h * input_w;
    int output_size = batch_size * out_channels * out_h * out_w;
    int weight_size = out_channels * in_channels * kernel_h * kernel_w;

    // Allocate device memory
    float *d_grad_output, *d_input, *d_weights;
    float *d_grad_input, *d_grad_weights, *d_grad_bias;

    CUDA_CHECK(cudaMalloc(&d_grad_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias, out_channels * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output, output_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, weight_size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks;

    // Compute grad_input
    blocks = (input_size + threads - 1) / threads;
    conv2d_backward_input_kernel<<<blocks, threads>>>(
        d_grad_output, d_weights, d_grad_input,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w, stride, padding
    );
    CUDA_CHECK_KERNEL();

    // Compute grad_weights
    blocks = (weight_size + threads - 1) / threads;
    conv2d_backward_weights_kernel<<<blocks, threads>>>(
        d_grad_output, d_input, d_grad_weights,
        batch_size, in_channels, out_channels,
        input_h, input_w, kernel_h, kernel_w, stride, padding
    );
    CUDA_CHECK_KERNEL();

    // Compute grad_bias
    blocks = (out_channels + threads - 1) / threads;
    conv2d_backward_bias_kernel<<<blocks, threads>>>(
        d_grad_output, d_grad_bias,
        batch_size, out_channels, out_h, out_w
    );
    CUDA_CHECK_KERNEL();

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(grad_input, d_grad_input, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_weights, d_grad_weights, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_bias, d_grad_bias, out_channels * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_grad_input));
    CUDA_CHECK(cudaFree(d_grad_weights));
    CUDA_CHECK(cudaFree(d_grad_bias));
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
    CUDA_CHECK(cudaMalloc(&d_params, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gradients, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_params, params, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gradients, gradients, size * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(d_params, d_gradients, learning_rate, size);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(params, d_params, size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_gradients));
}

} // extern "C"
