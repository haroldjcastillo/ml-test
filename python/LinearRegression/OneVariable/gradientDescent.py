import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    """Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function."""

    m = len(y);  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        h = X.dot(theta) - np.transpose([y])
        theta[0] = theta[0] - alpha * (1 / m) * X[:, 0].dot(h)
        theta[1] = theta[1] - alpha * (1 / m) * X[:, 1].dot(h)
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history
