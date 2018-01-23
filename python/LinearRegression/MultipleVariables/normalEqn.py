import numpy as np


def normalEqn(X, y):
    xT = np.transpose(X)
    return np.linalg.pinv(xT.dot(X)).dot(xT.dot(y))
