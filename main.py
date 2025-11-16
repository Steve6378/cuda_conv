#!/usr/bin/env python3
"""
CUDA Matrix Multiplication and Convolution Benchmarking Suite
This script compiles, runs, and benchmarks various CUDA implementations
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys


def run_command(cmd, description=""):
    """Run a shell command and return output"""
    if description:
        print(f"[{description}]")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def check_gpu():
    """Check GPU availability and CUDA version"""
    print("=" * 60)
    print("GPU and CUDA Version Check")
    print("=" * 60)

    nvidia_smi = run_command("nvidia-smi", "Checking GPU")
    print(nvidia_smi)

    nvcc_version = run_command("nvcc --version", "Checking CUDA version")
    print(nvcc_version)
    print()


def write_matrix_cpu():
    """Generate matrix_cpu.c source file"""
    code = """#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrixMultiplyCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * N * sizeof(float);

    float *A = (float*) malloc(size);
    float *B = (float*) malloc(size);
    float *C = (float*) malloc(size);

    srand(0);
    for (int i = 0; i < N * N; i++){
        A[i] = rand() % 100 / 100.0f;
        B[i] = rand() % 100 / 100.0f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matrixMultiplyCPU(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec)/1e9;

    printf("CPU execution time (N=%d): %f seconds\\n", N, elapsed);

    free(A); free(B); free(C);
    return 0;
}
"""
    with open("matrix_cpu.c", "w") as f:
        f.write(code)


def write_matrix_gpu_naive():
    """Generate matrix_gpu_naive.cu source file"""
    code = """#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplyGPU(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++){
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Naive GPU time (N=%d): %f ms\\n", N, ms);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
"""
    with open("matrix_gpu_naive.cu", "w") as f:
        f.write(code)


def write_matrix_gpu_tiled():
    """Generate matrix_gpu_tiled.cu source file"""
    code = """#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE - 1) / TILE; m++) {
        if (row < N && m * TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + m * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && m * TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(m * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++){
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Tiled GPU time (N=%d): %f ms\\n", N, ms);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
"""
    with open("matrix_gpu_tiled.cu", "w") as f:
        f.write(code)


def write_matrix_cublas():
    """Generate matrix_cublas.cu source file"""
    code = """#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = (size_t)N*N*sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N*N; i++){
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("cuBLAS time (N=%d): %f ms\\n", N, ms);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
"""
    with open("matrix_cublas.cu", "w") as f:
        f.write(code)


def compile_matrix_programs():
    """Compile all matrix multiplication programs"""
    print("=" * 60)
    print("Compiling Matrix Multiplication Programs")
    print("=" * 60)

    run_command("gcc matrix_cpu.c -O2 -o matrix_cpu", "Compiling CPU version")
    run_command("nvcc matrix_gpu_naive.cu -O2 -o matrix_gpu_naive", "Compiling naive GPU version")
    run_command("nvcc matrix_gpu_tiled.cu -O2 -o matrix_gpu_tiled", "Compiling tiled GPU version")
    run_command("nvcc matrix_cublas.cu -lcublas -O2 -o matrix_cublas", "Compiling cuBLAS version")

    print("All matrix programs compiled successfully!\n")


def compile_convolution_programs():
    """Compile convolution programs"""
    print("=" * 60)
    print("Compiling Convolution Programs")
    print("=" * 60)

    run_command("gcc benchmarks/conv_cpu.c -O2 -o conv_cpu", "Compiling CPU convolution")
    run_command("nvcc benchmarks/conv_gpu.cu -O2 -o conv_gpu", "Compiling GPU convolution")

    print("All convolution programs compiled successfully!\n")


def extract_ms(output):
    """Extract milliseconds from output and convert to seconds"""
    match = re.search(r"([0-9]+\.[0-9]+)\s*ms", output)
    if match:
        return float(match.group(1)) / 1000.0
    else:
        print(f"Output parsing failed:\n{output}")
        return float('nan')


def benchmark_matrix():
    """Run matrix multiplication benchmarks"""
    print("=" * 60)
    print("Running Matrix Multiplication Benchmarks")
    print("=" * 60)

    Ns = [256, 512, 1024, 1536, 2048]
    results = []

    for N in Ns:
        print(f"Running N={N}")

        cpu = float(subprocess.check_output(["./matrix_cpu", str(N)]).decode().split()[-2])

        out = subprocess.check_output(["./matrix_gpu_naive", str(N)]).decode()
        naive = extract_ms(out)

        out = subprocess.check_output(["./matrix_gpu_tiled", str(N)]).decode()
        tiled = extract_ms(out)

        out = subprocess.check_output(["./matrix_cublas", str(N)]).decode()
        cublas = extract_ms(out)

        results.append([N, cpu, naive, tiled, cublas])

    df = pd.DataFrame(results, columns=["N", "CPU_seconds", "Naive_seconds", "Tiled_seconds", "cuBLAS_seconds"])
    df.to_csv("results.csv", index=False)

    print("\nMatrix Multiplication Results:")
    print(df)
    print("\nResults saved to results.csv\n")

    return df


def benchmark_convolution():
    """Run convolution benchmarks"""
    print("=" * 60)
    print("Running Convolution Benchmarks")
    print("=" * 60)

    Ms = [256, 512, 1024]
    Ns = [3, 5, 7]
    conv_results = []

    for M in Ms:
        for N in Ns:
            print(f"Running convolution: M={M}, N={N}")

            # CPU convolution
            cpu_out = subprocess.check_output(["./conv_cpu", str(M), str(N)]).decode()
            cpu_match = re.search(r"([0-9]+\.[0-9]+)\s*seconds", cpu_out)
            cpu_time = float(cpu_match.group(1)) if cpu_match else 0

            # GPU convolution
            gpu_out = subprocess.check_output(["./conv_gpu", str(M), str(N)]).decode()
            gpu_match = re.search(r"([0-9]+\.[0-9]+)\s*ms", gpu_out)
            gpu_time = float(gpu_match.group(1)) / 1000.0 if gpu_match else 0

            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            conv_results.append([M, N, cpu_time, gpu_time, speedup])

    conv_df = pd.DataFrame(conv_results,
                          columns=["Image_Size_M", "Filter_Size_N", "CPU_seconds", "GPU_seconds", "Speedup"])
    conv_df.to_csv("convolution_results.csv", index=False)

    print("\nConvolution Results:")
    print(conv_df)
    print("\nResults saved to convolution_results.csv\n")

    return conv_df


def plot_matrix_speedup(df):
    """Plot matrix multiplication speedup"""
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["CPU_seconds"] / df["Naive_seconds"], marker="o", label="Naive CUDA Speedup")
    plt.plot(df["N"], df["CPU_seconds"] / df["Tiled_seconds"], marker="o", label="Tiled CUDA Speedup")
    plt.plot(df["N"], df["CPU_seconds"] / df["cuBLAS_seconds"], marker="o", label="cuBLAS Speedup")

    plt.xlabel("Matrix Size N")
    plt.ylabel("Speedup vs CPU")
    plt.title("GPU Speedup Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig("speedup_plot.png", dpi=100, bbox_inches='tight')
    print("Matrix speedup plot saved to speedup_plot.png")
    plt.close()


def plot_convolution_analysis(conv_df):
    """Plot convolution analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Speedup vs Image Size for different filter sizes
    for N in [3, 5, 7]:
        data = conv_df[conv_df["Filter_Size_N"] == N]
        ax1.plot(data["Image_Size_M"], data["Speedup"], marker="o", label=f"Filter Size {N}x{N}")

    ax1.set_xlabel("Image Size (M x M)")
    ax1.set_ylabel("Speedup (CPU/GPU)")
    ax1.set_title("Convolution Speedup vs Image Size")
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Execution time comparison
    df_512_5 = conv_df[(conv_df["Image_Size_M"] == 512) & (conv_df["Filter_Size_N"] == 5)]
    if not df_512_5.empty:
        ax2.bar(["CPU", "GPU"], [df_512_5["CPU_seconds"].values[0], df_512_5["GPU_seconds"].values[0]])
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Convolution Time (512x512 image, 5x5 filter)")
        ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig("convolution_analysis.png", dpi=100, bbox_inches='tight')
    print("Convolution analysis plot saved to convolution_analysis.png")
    plt.close()


def compile_shared_libraries():
    """Compile shared libraries"""
    print("=" * 60)
    print("Compiling Shared Libraries")
    print("=" * 60)

    run_command("nvcc -Xcompiler -fPIC -shared lib/matrix_lib.cu -O2 -o libmatrix.so",
                "Compiling matrix library")
    run_command("nvcc -Xcompiler -fPIC -shared lib/conv_lib.cu -O2 -o libconv.so",
                "Compiling convolution library")

    print("Shared libraries compiled successfully!\n")


def test_shared_libraries():
    """Test shared libraries"""
    print("=" * 60)
    print("Testing Shared Libraries")
    print("=" * 60)

    if os.path.exists("test_matrix_lib.py"):
        print("\n[Testing matrix library]")
        result = run_command("python3 test_matrix_lib.py")
        print(result)

    if os.path.exists("test_conv_lib.py"):
        print("\n[Testing convolution library]")
        result = run_command("python3 test_conv_lib.py")
        print(result)

    print()


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("CUDA Matrix Multiplication and Convolution Benchmark Suite")
    print("=" * 60 + "\n")

    # Check GPU availability
    check_gpu()

    # Generate source files if they don't exist
    if not os.path.exists("matrix_cpu.c"):
        print("Generating matrix_cpu.c...")
        write_matrix_cpu()

    if not os.path.exists("matrix_gpu_naive.cu"):
        print("Generating matrix_gpu_naive.cu...")
        write_matrix_gpu_naive()

    if not os.path.exists("matrix_gpu_tiled.cu"):
        print("Generating matrix_gpu_tiled.cu...")
        write_matrix_gpu_tiled()

    if not os.path.exists("matrix_cublas.cu"):
        print("Generating matrix_cublas.cu...")
        write_matrix_cublas()

    # Compile programs
    compile_matrix_programs()
    compile_convolution_programs()

    # Run benchmarks
    matrix_df = benchmark_matrix()
    conv_df = benchmark_convolution()

    # Generate plots
    print("=" * 60)
    print("Generating Plots")
    print("=" * 60)
    plot_matrix_speedup(matrix_df)
    plot_convolution_analysis(conv_df)
    print()

    # Compile and test shared libraries
    compile_shared_libraries()
    test_shared_libraries()

    print("=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results.csv (matrix multiplication results)")
    print("  - convolution_results.csv (convolution results)")
    print("  - speedup_plot.png (matrix speedup visualization)")
    print("  - convolution_analysis.png (convolution analysis)")
    print("  - libmatrix.so (matrix shared library)")
    print("  - libconv.so (convolution shared library)")
    print()


if __name__ == "__main__":
    main()
