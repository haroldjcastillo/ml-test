import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function."""

    m = len(y);  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        theta = theta - alpha * (np.transpose(X).dot(X.dot(theta) - np.transpose([y])) / m)
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history
