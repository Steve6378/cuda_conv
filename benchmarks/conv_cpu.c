#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU Convolution: applies an N×N filter to an M×M image
// Result is (M-N+1) × (M-N+1)
void convolutionCPU(unsigned int *image, unsigned int *filter, unsigned int *output, int M, int N) {
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

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;  // Image size
    int N = (argc > 2) ? atoi(argv[2]) : 5;    // Filter size (e.g., 3, 5, 7)

    size_t imageSize = M * M * sizeof(unsigned int);
    size_t filterSize = N * N * sizeof(unsigned int);
    int outputDim = M - N + 1;
    size_t outputSize = outputDim * outputDim * sizeof(unsigned int);

    unsigned int *image = (unsigned int*) malloc(imageSize);
    unsigned int *filter = (unsigned int*) malloc(filterSize);
    unsigned int *output = (unsigned int*) malloc(outputSize);

    // Initialize image with random values
    srand(0);
    for (int i = 0; i < M * M; i++) {
        image[i] = rand() % 256;  // Grayscale values 0-255
    }

    // Initialize filter (edge detection example - simple Sobel-like)
    for (int i = 0; i < N * N; i++) {
        filter[i] = 1;  // Simple averaging filter
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    convolutionCPU(image, filter, output, M, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec)/1e9;

    printf("CPU convolution time (M=%d, N=%d): %f seconds\n", M, N, elapsed);

    free(image); free(filter); free(output);
    return 0;
}
