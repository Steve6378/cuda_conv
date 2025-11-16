#!/usr/bin/env python3
"""
CNN from Scratch using CUDA
Build a simple Convolutional Neural Network using custom CUDA kernels
"""

import numpy as np
import ctypes


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


def main():
    """Test the CUDA CNN implementation"""
    print("=" * 60)
    print("CNN from Scratch using CUDA")
    print("=" * 60)
    print()

    # Initialize CNN
    print("Loading CUDA CNN library...")
    cnn = CudaCNN()
    print("Library loaded successfully!")
    print()

    # Initialize parameters
    print("Initializing network parameters...")
    params = initialize_params()
    print(f"  Conv1: {params['conv1_w'].shape} weights, {params['conv1_b'].shape} biases")
    print(f"  Conv2: {params['conv2_w'].shape} weights, {params['conv2_b'].shape} biases")
    print(f"  FC:    {params['fc_w'].shape} weights, {params['fc_b'].shape} biases")
    print()

    # Create dummy MNIST-like input (batch_size=2, channels=1, height=28, width=28)
    batch_size = 2
    x = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    print(f"Input shape: {x.shape}")
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

    print(f"Output shape: {output.shape}")
    print(f"Output (class probabilities):")
    print(output)
    print()

    # Verify probabilities sum to 1
    print("Verification:")
    for i in range(batch_size):
        prob_sum = output[i].sum()
        predicted_class = output[i].argmax()
        confidence = output[i].max()
        print(f"  Sample {i}: Predicted class = {predicted_class}, Confidence = {confidence:.4f}, Sum = {prob_sum:.6f}")

    print()
    print("=" * 60)
    print("CNN test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
