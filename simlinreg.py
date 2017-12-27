# coding=utf-8
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston

# Simple Linear Regression

#   y = β0 + β1 * x
#   β1 = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#   β0 = mean(y) – β1 * mean(x)

# Alternatives data sets
# data_set = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 4, 5, 6, 8, 9, 10, 13, 12]]
# data_set = [[1, 2, 4, 3, 5], [1, 3, 3, 2, 5]]
data_set = [[1, 2, 3], [1, 2, 3]]

x_mean = np.mean(data_set[0])
y_mean = np.mean(data_set[1])

x = np.subtract(data_set[:1], x_mean)
y = np.subtract(data_set[1:], y_mean)

x_y = np.multiply(x, y)
x_pow = np.power(x, 2)

b_1 = np.sum(x_y) / np.sum(x_pow)
b_0 = y_mean - b_1 * x_mean

print("β0: %f β1: %f" % (b_0, b_1))

predict = []
for i in data_set[0]:
    y = b_0 + (b_1 * i)
    predict.append(round(y, 2))

# Cost function J or Squared error function
# 1/2m * ∑(ÿ - (β0 + β1 * Xi))^2
# J(β0, β1)

# Alternatives predictions
# predict = [0.5, 1.0, 1.5]  # ~ β1 = 0.5
# predict = [0.0, 0.0, 0.0]  # ~ β1 = 0.0
# predict = [-0.5, -1.0, -1.5]  # ~ β1 = -0.5

error = np.power((np.subtract(predict, data_set[1])), 2)
J = np.sum(error) / (2 * len(data_set[1]))
print("Cost function: " + str(J))

# Root Mean Squared Error
#   RMSE = sqrt( sum( (pi – yi)^2 )/n )
rmse = math.sqrt(J)
print("Root Mean Squared Error: " + str(rmse))

plt.figure(1)
plt.subplot(211)
plt.scatter(data_set[0], data_set[1], marker='o', c='b')
plt.plot(data_set[0], predict, '--bo', color='green')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True, which='major', color='b', linestyle='-')

plt.subplot(212)
plt.scatter(predict[0], J, marker='x', c='r')
plt.xlabel('B0')
plt.ylabel('J(B0)')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()
