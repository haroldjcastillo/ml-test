# coding=utf-8

# Normal equation
# X^T = Transpose of matrix X
#   θ = (X^T * X)^1 * X^T * y

import numpy as np

X = np.matrix([[1, 2104, 5, 1, 45],
               [1, 1416, 3, 2, 40],
               [1, 1534, 3, 2, 30],
               [1, 852, 2, 1, 36]])
y = np.matrix([[460], [232], [315], [178]])

transposeX = np.transpose(X)
theta = np.linalg.pinv(transposeX * X) * transposeX * y
h = X * theta

print("X shape: ", X.shape, " Y shape", y.shape)
print("X: ", X, " y: ", y)
print("theta: ", theta)
print("h: ", h)
