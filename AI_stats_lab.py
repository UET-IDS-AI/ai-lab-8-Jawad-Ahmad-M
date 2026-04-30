"""
AI_stats_lab.py

Neural Networks Lab: 3-Layer Forward Pass and Backpropagation

Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    return (1 / (1 + np.exp(-z)))


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    h1 = sigmoid(X @ W1)
    h2 = sigmoid(h1 @ W2)
    y = sigmoid(h2 @ W3)
    
    return h1, h2, y




def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """
    m = X.shape[0]
    eps = 1e-8

    # -------- LOSS --------
    loss = -np.mean(label * np.log(y + eps) + (1 - label) * np.log(1 - y + eps))

    # -------- OUTPUT LAYER --------
    dZ3 = y - label                  # (m, 1)
    dW3 = (h2.T @ dZ3) / m           # (n_h2, 1)

    # -------- LAYER 2 --------
    dA2 = dZ3 @ W3.T                 # (m, n_h2)
    dZ2 = dA2 * h2 * (1 - h2)
    dW2 = (h1.T @ dZ2) / m           # (n_h1, n_h2)

    # -------- LAYER 1 --------
    dA1 = dZ2 @ W2.T                 # (m, n_h1)
    dZ1 = dA1 * h1 * (1 - h1)
    dW1 = (X.T @ dZ1) / m            # (n_x, n_h1)

    return dW1, dW2, dW3, loss
