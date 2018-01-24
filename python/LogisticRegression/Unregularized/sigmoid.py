import numpy as np


def sigmoid(z):
    """Compute the sigmoid of each value of z (z can be a matrix, vector or scalar)."""
    return 1 / (1 + np.exp(-z))
