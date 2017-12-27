# coding=utf-8
# The formula for the correlation (r) is
# r = (1/n-1) ( (∑x ∑y (x - ẍ) (y - ÿ) ) / Sx Sy)

import math
import numpy as np
from matplotlib import pyplot as plt

data_set = [[1, 2, 4, 3, 5], [1, 3, 3, 2, 5]]
# data_set = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 4, 5, 6, 8, 9, 10, 13, 12]]

print("X: ", data_set[0])
print("Y: ", data_set[1])

x_mean = np.mean(data_set[0])
y_mean = np.mean(data_set[1])
print("x mean: " + str(x_mean) + " y mean: " + str(y_mean))


def std(data):
    mean = np.mean(data)
    return math.sqrt(np.sum(np.power(np.subtract(data, mean), 2)) / (len(data) - 1))


def var(data):
    return (np.sum(np.power(data, 2)) / len(data)) - math.pow(np.mean(data), 2)


variance_x = var(data_set[0])
variance_y = var(data_set[1])
print("Variance x: " + str(variance_x) + " Variance y: " + str(variance_y))

x_y = np.multiply(data_set[0], data_set[1])
print("xy: ", x_y, " result: " + str(np.sum(x_y)))

# Covariance
# C(xy) = [(∑xy)/n] - (ẍ)(ÿ)
covariance = (np.sum(x_y) / len(data_set[0])) - (x_mean * y_mean)
print("Covariance: " + str(covariance))

r = covariance / (math.sqrt(variance_x) * math.sqrt(variance_y))
print("Correlation coeficient: " + str(r))

# y = B0 + B1 * x
# B1 = corr(x, y) * stdev(y) / stdev(x)

B1 = r * std(data_set[1]) / std(data_set[0])
print("Output value: " + str(B1))
