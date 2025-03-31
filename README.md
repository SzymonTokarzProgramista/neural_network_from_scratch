# neural_network_from_scratch

This project compares two implementations of a simple neural network trained on the MNIST dataset:

- **NumPy**: A network built from scratch, without any deep learning frameworks.
- **PyTorch**: A network built using PyTorch's high-level APIs for training and evaluation.

## Goals

- Understand how neural networks work by implementing one manually.
- Compare training time, accuracy, and learning behavior between the two approaches.
- Highlight the pros and cons of using a deep learning framework like PyTorch vs. building your own.

## Features

- Trains on the MNIST handwritten digits dataset.
- Manual forward and backward propagation in the NumPy version.
- PyTorch model uses `nn.Module`, loss functions, and optimizers.
- Side-by-side comparison of performance and metrics.
- Optional visualization of loss/accuracy curves.

## Requirements

- Python 3.x  
- NumPy  
- PyTorch  
- Matplotlib (optional, for plots)

You can install the dependencies using:

```bash
pip install numpy torch matplotlib
