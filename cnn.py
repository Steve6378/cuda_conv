#!/usr/bin/env python3
"""
CNN from Scratch using CUDA
Build a simple Convolutional Neural Network using custom CUDA kernels

NOTE: Currently only implements FORWARD PASS.
TODO: Need to add backprop kernels for actual training.
"""

import numpy as np
import ctypes
import gzip
import os
from urllib import request


class CudaCNN:
    """
    A simple CNN built from scratch using CUDA kernels
    Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC -> Softmax
    """

    def __init__(self, lib_path="./libcnn.so"):
        """Load the compiled CUDA library"""
        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Define ctypes argument types for all CUDA functions"""

        # Conv2D
        self.lib.cuda_conv2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # ReLU
        self.lib.cuda_relu.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]

        # MaxPool2D
        self.lib.cuda_maxpool2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # Fully Connected
        self.lib.cuda_fc.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # Softmax
        self.lib.cuda_softmax.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int
        ]

    def conv2d(self, input_tensor, weights, bias, stride=1, padding=0):
        """
        2D Convolution using CUDA

        Args:
            input_tensor: [batch, in_channels, height, width]
            weights: [out_channels, in_channels, kernel_h, kernel_w]
            bias: [out_channels]
            stride: convolution stride
            padding: zero padding
        Returns:
            output: [batch, out_channels, out_h, out_w]
        """
        batch, in_c, in_h, in_w = input_tensor.shape
        out_c, _, k_h, k_w = weights.shape

        out_h = (in_h + 2 * padding - k_h) // stride + 1
        out_w = (in_w + 2 * padding - k_w) // stride + 1

        output = np.zeros((batch, out_c, out_h, out_w), dtype=np.float32)

        self.lib.cuda_conv2d(
            input_tensor.ravel(), weights.ravel(), bias.ravel(), output.ravel(),
            batch, in_c, out_c, in_h, in_w, k_h, k_w, stride, padding
        )

        return output

    def relu(self, data):
        """ReLU activation using CUDA (in-place operation)"""
        data_copy = data.copy()
        self.lib.cuda_relu(data_copy.ravel(), data_copy.size)
        return data_copy

    def maxpool2d(self, input_tensor, pool_size=2, stride=2):
        """
        2D Max Pooling using CUDA

        Args:
            input_tensor: [batch, channels, height, width]
            pool_size: size of pooling window
            stride: pooling stride
        Returns:
            output: [batch, channels, out_h, out_w]
        """
        batch, channels, in_h, in_w = input_tensor.shape
        out_h = (in_h - pool_size) // stride + 1
        out_w = (in_w - pool_size) // stride + 1

        output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)

        self.lib.cuda_maxpool2d(
            input_tensor.ravel(), output.ravel(),
            batch, channels, in_h, in_w, pool_size, stride
        )

        return output

    def fc(self, input_tensor, weights, bias):
        """
        Fully connected layer using CUDA

        Args:
            input_tensor: [batch, in_features]
            weights: [out_features, in_features]
            bias: [out_features]
        Returns:
            output: [batch, out_features]
        """
        batch, in_feat = input_tensor.shape
        out_feat = weights.shape[0]

        output = np.zeros((batch, out_feat), dtype=np.float32)

        self.lib.cuda_fc(
            input_tensor.ravel(), weights.ravel(), bias.ravel(), output.ravel(),
            batch, in_feat, out_feat
        )

        return output

    def softmax(self, data):
        """Softmax activation using CUDA"""
        data_copy = data.copy()
        batch, num_classes = data_copy.shape
        self.lib.cuda_softmax(data_copy.ravel(), batch, num_classes)
        return data_copy

    def forward(self, x, params):
        """
        Forward pass through a simple CNN
        Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> FC -> Softmax

        Args:
            x: Input tensor [batch, 1, 28, 28] for MNIST-like data
            params: Dictionary of weights and biases
        Returns:
            output: [batch, num_classes] class probabilities
        """
        # Layer 1: Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2)
        x = self.conv2d(x, params['conv1_w'], params['conv1_b'], stride=1, padding=1)
        x = self.relu(x)
        x = self.maxpool2d(x, pool_size=2, stride=2)

        # Layer 2: Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
        x = self.conv2d(x, params['conv2_w'], params['conv2_b'], stride=1, padding=1)
        x = self.relu(x)
        x = self.maxpool2d(x, pool_size=2, stride=2)

        # Flatten
        batch = x.shape[0]
        x = x.reshape(batch, -1)

        # Fully connected + Softmax
        x = self.fc(x, params['fc_w'], params['fc_b'])
        x = self.softmax(x)

        return x


def initialize_params():
    """Initialize random weights for a simple CNN (MNIST-like architecture)"""
    np.random.seed(42)

    params = {
        # Conv1: 1 input channel, 32 output channels, 3x3 kernel
        'conv1_w': np.random.randn(32, 1, 3, 3).astype(np.float32) * 0.1,
        'conv1_b': np.zeros(32, dtype=np.float32),

        # Conv2: 32 input channels, 64 output channels, 3x3 kernel
        'conv2_w': np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.1,
        'conv2_b': np.zeros(64, dtype=np.float32),

        # FC: After 2 poolings, 28x28 -> 14x14 -> 7x7, with 64 channels = 7*7*64 = 3136 features
        'fc_w': np.random.randn(10, 3136).astype(np.float32) * 0.1,
        'fc_b': np.zeros(10, dtype=np.float32),
    }

    return params


def load_mnist(data_dir='./data'):
    """Download and load MNIST dataset"""
    os.makedirs(data_dir, exist_ok=True)

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    # Download files if not present
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            request.urlretrieve(base_url + filename, filepath)

    # Load training data
    with gzip.open(os.path.join(data_dir, files['train_images']), 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(os.path.join(data_dir, files['train_labels']), 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Load test data
    with gzip.open(os.path.join(data_dir, files['test_images']), 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(os.path.join(data_dir, files['test_labels']), 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Normalize to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    print(f"Loaded MNIST: {len(train_images)} training, {len(test_images)} test images")

    return (train_images, train_labels), (test_images, test_labels)


def main():
    """Test the CUDA CNN implementation with MNIST data"""
    print("=" * 60)
    print("CNN from Scratch using CUDA")
    print("=" * 60)
    print()

    # Load MNIST data
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    print()

    # Initialize CNN
    print("Loading CUDA CNN library...")
    cnn = CudaCNN()
    print("Library loaded successfully!")
    print()

    # Initialize parameters (random weights - NOT TRAINED!)
    print("Initializing network parameters (random weights)...")
    params = initialize_params()
    print(f"  Conv1: {params['conv1_w'].shape} weights, {params['conv1_b'].shape} biases")
    print(f"  Conv2: {params['conv2_w'].shape} weights, {params['conv2_b'].shape} biases")
    print(f"  FC:    {params['fc_w'].shape} weights, {params['fc_b'].shape} biases")
    print()

    # Test on a small batch of real MNIST images
    batch_size = 8
    x = test_images[:batch_size]
    y = test_labels[:batch_size]
    print(f"Testing on {batch_size} real MNIST images")
    print(f"True labels: {y}")
    print()

    # Forward pass
    print("Running forward pass through CNN...")
    print("  [1/5] Conv1 + ReLU + MaxPool")
    print("  [2/5] Conv2 + ReLU + MaxPool")
    print("  [3/5] Flatten")
    print("  [4/5] Fully Connected")
    print("  [5/5] Softmax")
    output = cnn.forward(x, params)
    print()

    # Show predictions
    predictions = output.argmax(axis=1)
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {(predictions == y).mean() * 100:.1f}% (should be ~10% for random weights)")
    print()

    # Show confidence for each sample
    print("Detailed predictions:")
    for i in range(batch_size):
        predicted = predictions[i]
        true_label = y[i]
        confidence = output[i, predicted]
        correct = "✓" if predicted == true_label else "✗"
        print(f"  Sample {i}: Predicted={predicted}, True={true_label}, Confidence={confidence:.4f} {correct}")

    print()
    print("=" * 60)
    print("LIMITATIONS:")
    print("- Forward pass only (no backprop implemented)")
    print("- Random weights (no training implemented)")
    print("- Need to add: Conv/Pool/ReLU backward kernels")
    print("- Need to add: Proper training loop with SGD")
    print("=" * 60)


if __name__ == "__main__":
    main()
