import ctypes
import numpy as np
import time

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libconv.so")

# Define argument types for GPU convolution
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

# Define argument types for CPU convolution
lib.cpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

# Test parameters
M = 512  # Image size
N = 5    # Filter size

# Create test image and filter
np.random.seed(0)
image = np.random.randint(0, 256, (M, M), dtype=np.uint32)

# Simple blur filter (all positive values)
filter_edge = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
], dtype=np.uint32)

outputDim = M - N + 1
output_cpu = np.zeros((outputDim, outputDim), dtype=np.uint32)
output_gpu = np.zeros((outputDim, outputDim), dtype=np.uint32)

# Test CPU convolution
print("Testing CPU convolution...")
start = time.time()
lib.cpu_convolution(image.ravel(), filter_edge.ravel(), output_cpu.ravel(), M, N)
cpu_time = time.time() - start
print(f"CPU convolution completed in {cpu_time:.4f} seconds")

# Test GPU convolution
print("\nTesting GPU convolution...")
start = time.time()
lib.gpu_convolution(image.ravel(), filter_edge.ravel(), output_gpu.ravel(), M, N)
gpu_time = time.time() - start
print(f"GPU convolution completed in {gpu_time:.4f} seconds")

# Calculate speedup
speedup = cpu_time / gpu_time
print(f"\nSpeedup: {speedup:.2f}x")

# Show sample output
print(f"\nCPU output sample (first 5x5):\n{output_cpu[:5, :5]}")
print(f"\nGPU output sample (first 5x5):\n{output_gpu[:5, :5]}")

# Verify results match
if np.allclose(output_cpu, output_gpu):
    print("\n✓ CPU and GPU results match!")
else:
    print("\n✗ Warning: CPU and GPU results differ")
