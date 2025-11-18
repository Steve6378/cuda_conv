# CUDA Convolutional Neural Network

A high-performance Convolutional Neural Network (CNN) implementation from scratch using custom CUDA kernels. This project demonstrates GPU-accelerated deep learning with full forward and backward propagation for training.

## Overview

This project implements a complete CNN training pipeline using CUDA C/C++ for performance-critical operations:
- **Custom CUDA kernels** for convolution, pooling, activation functions, and fully-connected layers
- **Full backpropagation** with gradient computation for all layers
- **Training capabilities** with mini-batch gradient descent
- **MNIST dataset support** with automatic downloading
- **Comprehensive benchmarking** suite comparing CPU vs GPU performance

### Architecture

```
Input (28x28x1) → Conv(5x5, 8 filters) → ReLU → MaxPool(2x2) →
Conv(5x5, 16 filters) → ReLU → MaxPool(2x2) → Flatten →
Fully Connected(10 classes) → Softmax
```

## Prerequisites

- **CUDA Toolkit** (tested with CUDA 11.0+)
- **NVIDIA GPU** with compute capability 3.0 or higher
- **Python 3.7+**
- **GCC/G++** compiler
- **nvcc** CUDA compiler

### Check CUDA Installation

```bash
nvcc --version
nvidia-smi
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cuda_conv
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Compile CUDA libraries**
```bash
make
```

Or compile manually:
```bash
# Compile CNN library
nvcc -o libcnn.so -shared -Xcompiler -fPIC lib/cnn_lib.cu

# Compile convolution benchmarks
nvcc -o libconv.so -shared -Xcompiler -fPIC lib/conv_lib.cu

# Compile matrix multiplication benchmarks
nvcc -o libmatrix.so -shared -Xcompiler -fPIC lib/matrix_lib.cu
```

## Project Structure

```
cuda_conv/
├── cnn.py                  # Main CNN implementation with training
├── benchmarks.py           # Performance benchmarking suite
├── requirements.txt        # Python dependencies
├── lib/                    # CUDA source files
│   ├── cnn_lib.cu         # CNN CUDA kernels (conv, pool, FC, backprop)
│   ├── conv_lib.cu        # Convolution benchmarks
│   └── matrix_lib.cu      # Matrix multiplication benchmarks
├── tests/                  # Unit tests
│   ├── test_conv_lib.py   # Convolution tests
│   └── test_matrix_lib.py # Matrix multiplication tests
└── benchmarks/            # Generated benchmark data (gitignored)
```

## Usage

### Training a CNN on MNIST

```python
from cnn import CudaCNN, load_mnist

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Initialize CNN
cnn = CudaCNN(lib_path="./libcnn.so")

# Train the model
cnn.train(
    X_train[:1000],
    y_train[:1000],
    epochs=5,
    batch_size=32,
    learning_rate=0.01
)

# Evaluate
accuracy = cnn.evaluate(X_test[:100], y_test[:100])
print(f"Test Accuracy: {accuracy:.2%}")
```

### Running Benchmarks

```bash
# Run full benchmark suite
python benchmarks.py

# Results saved to:
# - results.csv (matrix multiplication benchmarks)
# - convolution_results.csv (convolution benchmarks)
# - speedup_plot.png (visualization)
```

### Running Tests

```bash
# Test convolution operations
python tests/test_conv_lib.py

# Test matrix operations
python tests/test_matrix_lib.py
```

## Performance

The CUDA implementation provides significant speedups over CPU:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Conv 28x28 | ~500ms | ~5ms | ~100x |
| MaxPool | ~50ms | ~1ms | ~50x |
| Matrix Mult (1024x1024) | ~2000ms | ~20ms | ~100x |

*Benchmarks run on NVIDIA RTX 3080*

## CUDA Kernels Implemented

### Forward Pass
- `cuda_conv2d` - 2D convolution with shared memory optimization
- `cuda_relu` - ReLU activation
- `cuda_maxpool2d` - 2x2 max pooling
- `cuda_fully_connected` - Dense layer
- `cuda_softmax` - Softmax activation with numerical stability

### Backward Pass
- `cuda_conv2d_backward` - Convolution gradient computation
- `cuda_relu_backward` - ReLU gradient
- `cuda_maxpool2d_backward` - Max pooling gradient with index tracking
- `cuda_fully_connected_backward` - Dense layer gradients
- `cuda_softmax_cross_entropy_backward` - Combined softmax+CE gradient

## Development

### Modifying CUDA Kernels

1. Edit the `.cu` files in the `lib/` directory
2. Recompile: `make` or manually compile with `nvcc`
3. Run tests to verify: `python tests/test_*.py`

### Adding New Layers

1. Implement forward and backward CUDA kernels in `lib/cnn_lib.cu`
2. Add Python wrapper methods in `cnn.py`
3. Add tests in `tests/`

## Troubleshooting

### CUDA Library Not Found
```
Error: libcnn.so: cannot open shared object file
```
**Solution**: Compile the CUDA code first with `make` or manually with `nvcc`

### GPU Memory Errors
```
Error: out of memory
```
**Solution**: Reduce batch size or image dimensions

### Import Errors
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

## References

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## License

This project is for educational purposes.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Authors

Built as part of a GPU computing and deep learning project.
