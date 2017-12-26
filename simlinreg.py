# coding=utf-8
# Simple Linear Regression
#   y = B0 + B1 * x
#   B1 = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#   B0 = mean(y) – B1 * mean(x)
# Root Mean Squared Error
#   RMSE = sqrt( sum( (pi – yi)^2 )/n )

import numpy as np
from matplotlib import pyplot as plt

# data_set = [[1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83],
#            [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]]

data_set = [[1, 2, 4, 3, 5], [1, 3, 3, 2, 5]]

x_mean = np.mean(data_set[0])
y_mean = np.mean(data_set[1])

x = np.subtract(data_set[:1], x_mean)
y = np.subtract(data_set[1:], y_mean)

x_y = np.multiply(x, y)
x_pow = np.power(x, 2)

b_1 = np.sum(x_y) / np.sum(x_pow)
b_0 = y_mean - b_1 * x_mean

print("B0: %f B1: %f" % (b_0, b_1))

predict = []
for i in data_set[0]:
    y = b_0 + (b_1 * i)
    predict.append(round(y, 2))

print(predict)

plt.scatter(data_set[0], data_set[1], marker='o', c='b')
plt.plot(predict, data_set[0])
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()
