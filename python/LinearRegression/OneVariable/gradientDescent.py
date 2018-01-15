import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y);  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(0, num_iters):
        h = np.dot(X, theta) - np.array(y)
        theta[0] = theta[0] - alpha * (1 / m) * np.sum(h * X[:, 0])
        theta[1] = theta[1] - alpha * (1 / m) * np.sum(h * X[:, 1])
        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history