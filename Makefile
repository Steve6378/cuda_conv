# Makefile for CUDA CNN Project
# Compiles all CUDA libraries needed for the CNN implementation

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O3 -Xcompiler -fPIC -shared

# Library names
CNN_LIB = libcnn.so
CONV_LIB = libconv.so
MATRIX_LIB = libmatrix.so

# Source files
CNN_SRC = lib/cnn_lib.cu
CONV_SRC = lib/conv_lib.cu
MATRIX_SRC = lib/matrix_lib.cu

# Default target - build all libraries
all: $(CNN_LIB) $(CONV_LIB) $(MATRIX_LIB)
	@echo "All libraries compiled successfully!"
	@echo "Run 'python cnn.py' to train the CNN"
	@echo "Run 'python benchmarks.py' to run performance benchmarks"

# Build CNN library (main library for training)
$(CNN_LIB): $(CNN_SRC)
	@echo "Compiling CNN library..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $(CNN_LIB) built successfully"

# Build convolution benchmark library
$(CONV_LIB): $(CONV_SRC)
	@echo "Compiling convolution benchmark library..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $(CONV_LIB) built successfully"

# Build matrix multiplication benchmark library
$(MATRIX_LIB): $(MATRIX_SRC)
	@echo "Compiling matrix multiplication benchmark library..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✓ $(MATRIX_LIB) built successfully"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(CNN_LIB) $(CONV_LIB) $(MATRIX_LIB)
	rm -f *.o
	rm -rf __pycache__
	rm -f tests/__pycache__
	@echo "✓ Clean complete"

# Test the compiled libraries
test: all
	@echo "Running tests..."
	python tests/test_matrix_lib.py
	python tests/test_conv_lib.py
	@echo "✓ All tests passed"

# Run benchmarks
benchmark: all
	@echo "Running performance benchmarks..."
	python benchmarks.py
	@echo "✓ Benchmarks complete"

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc > /dev/null || (echo "ERROR: nvcc not found. Install CUDA Toolkit first." && exit 1)
	@nvidia-smi > /dev/null || (echo "ERROR: nvidia-smi failed. Check GPU drivers." && exit 1)
	@echo "✓ CUDA Toolkit found:"
	@nvcc --version | grep "release"
	@echo "✓ GPU detected:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✓ Python dependencies installed"

# Full setup (check CUDA, install deps, build)
setup: check-cuda install-deps all
	@echo "✓ Setup complete! Ready to train."

# Help message
help:
	@echo "CUDA CNN Project - Makefile Commands"
	@echo "===================================="
	@echo "make              - Build all CUDA libraries"
	@echo "make clean        - Remove build artifacts"
	@echo "make test         - Build and run tests"
	@echo "make benchmark    - Build and run performance benchmarks"
	@echo "make check-cuda   - Verify CUDA installation"
	@echo "make install-deps - Install Python dependencies"
	@echo "make setup        - Full project setup"
	@echo "make help         - Show this help message"

.PHONY: all clean test benchmark check-cuda install-deps setup help
