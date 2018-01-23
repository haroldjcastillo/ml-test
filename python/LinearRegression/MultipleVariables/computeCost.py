import numpy as np


def computeCost(X, y, theta):
    """computes the cost of using theta as the parameter for linear regression to fit the data points in X and y"""

    m = len(y)  # number of training examples
    prediction = X.dot(theta)
    sqr_error = np.power(prediction - np.transpose([y]), 2)
    J = sqr_error.sum(axis=0) / (2 * m)

    return J
