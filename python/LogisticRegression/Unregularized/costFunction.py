from sigmoid import sigmoid
import numpy as np


def costFunction(theta, X, y):
    m = len(y)
    sgmd = 1 / (1 + np.exp(-np.dot(X, theta)))
    J = - (np.sum(y * np.log(sgmd).T + (1 - y) * np.log(1 - sgmd).T) / m)

    grad = (1. / m) * np.dot(sgmd.T - y, X).T

    return J, grad
