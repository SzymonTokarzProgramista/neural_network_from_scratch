import numpy as np
from tensorflow.keras.datasets import mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-8), axis=1))

def mse(pred, target):
    return np.mean((pred - target) ** 2)

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def apply_dropout(x, rate):
    mask = (np.random.rand(*x.shape) > rate).astype(float)
    return x * mask, mask

def train_numpy():
    # Dane
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # Parametry
    input_size, hidden_size, output_size = 784, 64, 10
    lr, epochs, batch_size, dropout_rate = 0.1, 10, 64, 0.3

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
    b2 = np.zeros((1, output_size))

    losses, mses = [], []

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train_oh[idx]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            z1 = np.dot(X_batch, W1) + b1
            a1 = sigmoid(z1)
            a1, mask = apply_dropout(a1, dropout_rate)
            z2 = np.dot(a1, W2) + b2
            a2 = softmax(z2)

            dz2 = a2 - y_batch
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1) * mask
            dW1 = np.dot(X_batch.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W2 -= lr * dW2 / batch_size
            b2 -= lr * db2 / batch_size
            W1 -= lr * dW1 / batch_size
            b1 -= lr * db1 / batch_size

        # Ewaluacja po każdej epoce
        z1 = np.dot(X_test, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        loss = cross_entropy(a2, y_test_oh)
        mse_val = mse(a2, y_test_oh)

        losses.append(loss)
        mses.append(mse_val)

    return losses, mses
