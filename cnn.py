#!/usr/bin/env python3
"""
CNN from Scratch using CUDA
Build a simple Convolutional Neural Network using custom CUDA kernels

Full implementation with forward AND backward passes for training!
"""

import numpy as np
import ctypes
import gzip
import os
import time
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

        # Conv2D forward
        self.lib.cuda_conv2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # Conv2D backward
        self.lib.cuda_conv2d_backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # ReLU forward
        self.lib.cuda_relu.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]

        # ReLU backward
        self.lib.cuda_relu_backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]

        # MaxPool2D forward
        self.lib.cuda_maxpool2d.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # MaxPool2D backward
        self.lib.cuda_maxpool2d_backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int
        ]

        # Fully Connected forward
        self.lib.cuda_fc.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # Fully Connected backward
        self.lib.cuda_fc_backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        # Softmax forward
        self.lib.cuda_softmax.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int
        ]

        # Softmax + cross-entropy backward
        self.lib.cuda_softmax_cross_entropy_backward.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int
        ]

        # Cross-entropy loss
        self.lib.cuda_cross_entropy_loss.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int
        ]

        # SGD update
        self.lib.cuda_sgd_update.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_float, ctypes.c_int
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

    def compute_loss(self, predictions, labels):
        """Compute cross-entropy loss"""
        batch_size, num_classes = predictions.shape
        labels_int = labels.astype(np.int32)
        loss = np.zeros(1, dtype=np.float32)

        self.lib.cuda_cross_entropy_loss(
            predictions.ravel(), labels_int, loss,
            batch_size, num_classes
        )
        return loss[0]

    def sgd_update(self, params, gradients, learning_rate):
        """Update parameters using SGD"""
        self.lib.cuda_sgd_update(
            params.ravel(), gradients.ravel(),
            learning_rate, params.size
        )

    def forward(self, x, params, save_cache=False):
        """
        Forward pass through a simple CNN
        Architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> FC -> Softmax

        Args:
            x: Input tensor [batch, 1, 28, 28] for MNIST-like data
            params: Dictionary of weights and biases
            save_cache: If True, save intermediate activations for backprop
        Returns:
            output: [batch, num_classes] class probabilities
            cache: (optional) dictionary of intermediate values for backprop
        """
        cache = {}  if save_cache else None

        # Layer 1: Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2)
        if save_cache:
            cache['input'] = x.copy()

        conv1 = self.conv2d(x, params['conv1_w'], params['conv1_b'], stride=1, padding=1)
        if save_cache:
            cache['conv1'] = conv1.copy()

        relu1 = self.relu(conv1)
        if save_cache:
            cache['relu1'] = relu1.copy()

        pool1 = self.maxpool2d(relu1, pool_size=2, stride=2)
        if save_cache:
            cache['pool1'] = pool1.copy()

        # Layer 2: Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2)
        conv2 = self.conv2d(pool1, params['conv2_w'], params['conv2_b'], stride=1, padding=1)
        if save_cache:
            cache['conv2'] = conv2.copy()

        relu2 = self.relu(conv2)
        if save_cache:
            cache['relu2'] = relu2.copy()

        pool2 = self.maxpool2d(relu2, pool_size=2, stride=2)
        if save_cache:
            cache['pool2'] = pool2.copy()

        # Flatten
        batch = pool2.shape[0]
        flattened = pool2.reshape(batch, -1)
        if save_cache:
            cache['flattened'] = flattened.copy()
            cache['pool2_shape'] = pool2.shape

        # Fully connected + Softmax
        fc_out = self.fc(flattened, params['fc_w'], params['fc_b'])
        if save_cache:
            cache['fc_out'] = fc_out.copy()

        softmax_out = self.softmax(fc_out)

        if save_cache:
            return softmax_out, cache
        return softmax_out

    def backward(self, predictions, labels, params, cache):
        """
        Backward pass to compute gradients

        Args:
            predictions: Softmax output [batch, num_classes]
            labels: True labels [batch]
            params: Current parameters
            cache: Intermediate activations from forward pass
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        batch_size = predictions.shape[0]
        labels_int = labels.astype(np.int32)
        grads = {}

        # Backward through softmax + cross-entropy
        grad_fc_out = np.zeros_like(predictions)
        self.lib.cuda_softmax_cross_entropy_backward(
            predictions.ravel(), labels_int, grad_fc_out.ravel(),
            batch_size, 10
        )

        # Backward through FC layer
        grad_flattened = np.zeros_like(cache['flattened'])
        grads['fc_w'] = np.zeros_like(params['fc_w'])
        grads['fc_b'] = np.zeros_like(params['fc_b'])

        self.lib.cuda_fc_backward(
            grad_fc_out.ravel(), cache['flattened'].ravel(), params['fc_w'].ravel(),
            grad_flattened.ravel(), grads['fc_w'].ravel(), grads['fc_b'].ravel(),
            batch_size, cache['flattened'].shape[1], 10
        )

        # Reshape back to pool2 shape
        grad_pool2 = grad_flattened.reshape(cache['pool2_shape'])

        # Backward through MaxPool2
        grad_relu2 = np.zeros_like(cache['relu2'])
        pool_size, stride = 2, 2
        self.lib.cuda_maxpool2d_backward(
            grad_pool2.ravel(), cache['relu2'].ravel(), cache['pool2'].ravel(),
            grad_relu2.ravel(),
            batch_size, 64, cache['relu2'].shape[2], cache['relu2'].shape[3],
            pool_size, stride
        )

        # Backward through ReLU2
        grad_conv2 = np.zeros_like(cache['conv2'])
        self.lib.cuda_relu_backward(
            grad_relu2.ravel(), cache['conv2'].ravel(), grad_conv2.ravel(),
            grad_conv2.size
        )

        # Backward through Conv2
        grad_pool1 = np.zeros_like(cache['pool1'])
        grads['conv2_w'] = np.zeros_like(params['conv2_w'])
        grads['conv2_b'] = np.zeros_like(params['conv2_b'])

        self.lib.cuda_conv2d_backward(
            grad_conv2.ravel(), cache['pool1'].ravel(), params['conv2_w'].ravel(),
            grad_pool1.ravel(), grads['conv2_w'].ravel(), grads['conv2_b'].ravel(),
            batch_size, 32, 64,
            cache['pool1'].shape[2], cache['pool1'].shape[3], 3, 3, 1, 1
        )

        # Backward through MaxPool1
        grad_relu1 = np.zeros_like(cache['relu1'])
        self.lib.cuda_maxpool2d_backward(
            grad_pool1.ravel(), cache['relu1'].ravel(), cache['pool1'].ravel(),
            grad_relu1.ravel(),
            batch_size, 32, cache['relu1'].shape[2], cache['relu1'].shape[3],
            pool_size, stride
        )

        # Backward through ReLU1
        grad_conv1 = np.zeros_like(cache['conv1'])
        self.lib.cuda_relu_backward(
            grad_relu1.ravel(), cache['conv1'].ravel(), grad_conv1.ravel(),
            grad_conv1.size
        )

        # Backward through Conv1
        grad_input = np.zeros_like(cache['input'])
        grads['conv1_w'] = np.zeros_like(params['conv1_w'])
        grads['conv1_b'] = np.zeros_like(params['conv1_b'])

        self.lib.cuda_conv2d_backward(
            grad_conv1.ravel(), cache['input'].ravel(), params['conv1_w'].ravel(),
            grad_input.ravel(), grads['conv1_w'].ravel(), grads['conv1_b'].ravel(),
            batch_size, 1, 32,
            cache['input'].shape[2], cache['input'].shape[3], 3, 3, 1, 1
        )

        return grads

    def train_step(self, x_batch, y_batch, params, learning_rate):
        """
        Single training step: forward + backward + update

        Args:
            x_batch: Input batch [batch, 1, 28, 28]
            y_batch: Labels [batch]
            params: Current parameters
            learning_rate: Learning rate for SGD
        Returns:
            loss: Training loss for this batch
        """
        # Forward pass with caching
        predictions, cache = self.forward(x_batch, params, save_cache=True)

        # Compute loss
        loss = self.compute_loss(predictions, y_batch)

        # Backward pass
        grads = self.backward(predictions, y_batch, params, cache)

        # Update parameters
        for key in params:
            self.sgd_update(params[key], grads[key], learning_rate)

        return loss


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


def evaluate(cnn, images, labels, params, batch_size=100):
    """Evaluate accuracy on a dataset"""
    n_samples = len(images)
    correct = 0

    for i in range(0, n_samples, batch_size):
        batch_x = images[i:i+batch_size]
        batch_y = labels[i:i+batch_size]

        predictions = cnn.forward(batch_x, params, save_cache=False)
        pred_labels = predictions.argmax(axis=1)
        correct += (pred_labels == batch_y).sum()

    return correct / n_samples


def main():
    """Train a CUDA CNN on MNIST"""
    print("=" * 70)
    print(" " * 20 + "CUDA CNN Training")
    print("=" * 70)
    print()

    # Load MNIST data
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist()
    print(f"  Train: {len(train_images)} images")
    print(f"  Test:  {len(test_images)} images")
    print()

    # Initialize CNN
    print("Loading CUDA CNN library...")
    cnn = CudaCNN()
    print("  Library loaded successfully!")
    print()

    # Initialize parameters
    print("Initializing network parameters...")
    params = initialize_params()
    print(f"  Conv1: {params['conv1_w'].shape}")
    print(f"  Conv2: {params['conv2_w'].shape}")
    print(f"  FC:    {params['fc_w'].shape}")
    print()

    # Training hyperparameters
    epochs = 3
    batch_size = 32
    learning_rate = 0.01
    print_every = 100

    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training batches per epoch: {len(train_images) // batch_size}")
    print()

    # Evaluate before training
    print("Evaluating on test set before training...")
    test_acc = evaluate(cnn, test_images[:1000], test_labels[:1000], params)
    print(f"  Initial test accuracy: {test_acc * 100:.2f}%")
    print()

    print("=" * 70)
    print("Starting Training")
    print("=" * 70)
    print()

    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]

        print(f"Epoch {epoch + 1}/{epochs}")

        for i in range(0, len(train_images), batch_size):
            batch_x = train_images_shuffled[i:i+batch_size]
            batch_y = train_labels_shuffled[i:i+batch_size]

            # Skip incomplete batches
            if len(batch_x) < batch_size:
                continue

            # Training step
            loss = cnn.train_step(batch_x, batch_y, params, learning_rate)
            epoch_loss += loss
            n_batches += 1

            # Print progress
            if n_batches % print_every == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Batch {n_batches:4d}: Loss = {avg_loss:.4f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s, Avg Loss = {avg_loss:.4f}")

        # Evaluate on test set
        test_acc = evaluate(cnn, test_images[:1000], test_labels[:1000], params)
        print(f"  Test accuracy: {test_acc * 100:.2f}%")
        print()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()

    # Final evaluation on full test set
    print("Final evaluation on full test set...")
    final_acc = evaluate(cnn, test_images, test_labels, params)
    print(f"  Final test accuracy: {final_acc * 100:.2f}%")
    print()

    # Show some predictions
    print("Sample predictions:")
    sample_idx = np.random.choice(len(test_images), 10, replace=False)
    sample_x = test_images[sample_idx]
    sample_y = test_labels[sample_idx]

    predictions = cnn.forward(sample_x, params, save_cache=False)
    pred_labels = predictions.argmax(axis=1)

    for i in range(10):
        pred = pred_labels[i]
        true = sample_y[i]
        conf = predictions[i, pred]
        status = "✓" if pred == true else "✗"
        print(f"  {status} Predicted: {pred}, True: {true}, Confidence: {conf:.3f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
