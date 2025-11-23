#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;  // Image size
    int N = (argc > 2) ? atoi(argv[2]) : 5;    // Filter size

    size_t imageSize = M * M * sizeof(unsigned int);
    size_t filterSize = N * N * sizeof(unsigned int);
    int outputDim = M - N + 1;
    size_t outputSize = outputDim * outputDim * sizeof(unsigned int);

    unsigned int *h_image = (unsigned int*) malloc(imageSize);
    unsigned int *h_filter = (unsigned int*) malloc(filterSize);
    unsigned int *h_output = (unsigned int*) malloc(outputSize);

    // Initialize image and filter
    srand(0);
    for (int i = 0; i < M * M; i++) {
        h_image[i] = rand() % 256;
    }
    for (int i = 0; i < N * N; i++) {
        h_filter[i] = 1;
    }

    unsigned int *d_image, *d_filter, *d_output;
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outputDim + 15) / 16, (outputDim + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolutionGPU<<<grid, block>>>(d_image, d_filter, d_output, M, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("GPU convolution time (M=%d, N=%d): %f ms\n", M, N, ms);

    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image); cudaFree(d_filter); cudaFree(d_output);
    free(h_image); free(h_filter); free(h_output);

    return 0;
}
