#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Convolution kernel
__global__ void convolutionGPU(unsigned int *image, unsigned int *filter, unsigned int *output, int M, int N) {
    int outputSize = M - N + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < outputSize && col < outputSize) {
        unsigned int sum = 0;
        for (int fi = 0; fi < N; fi++) {
            for (int fj = 0; fj < N; fj++) {
                sum += image[(row + fi) * M + (col + fj)] * filter[fi * N + fj];
            }
        }
        output[row * outputSize + col] = sum;
    }
}

// Exposed C function for GPU convolution
extern "C" void gpu_convolution(unsigned int *h_image, unsigned int *h_filter, unsigned int *h_output, int M, int N) {
    size_t imageSize = M * M * sizeof(unsigned int);
    size_t filterSize = N * N * sizeof(unsigned int);
    int outputDim = M - N + 1;
    size_t outputSize = outputDim * outputDim * sizeof(unsigned int);

    unsigned int *d_image, *d_filter, *d_output;
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outputDim + 15) / 16, (outputDim + 15) / 16);

    convolutionGPU<<<grid, block>>>(d_image, d_filter, d_output, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image); cudaFree(d_filter); cudaFree(d_output);
}

// Exposed C function for CPU convolution
extern "C" void cpu_convolution(unsigned int *image, unsigned int *filter, unsigned int *output, int M, int N) {
    int outputSize = M - N + 1;

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            unsigned int sum = 0;
            for (int fi = 0; fi < N; fi++) {
                for (int fj = 0; fj < N; fj++) {
                    sum += image[(i + fi) * M + (j + fj)] * filter[fi * N + fj];
                }
            }
            output[i * outputSize + j] = sum;
        }
    }
}
