# coding=utf-8
# Simple Linear Regression
#   y = B0 + B1 * x
#   B1 = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#   B0 = mean(y) – B1 * mean(x)

import math
import numpy as np
from matplotlib import pyplot as plt

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

plt.scatter(data_set[0], data_set[1], marker='o', c='b')
plt.plot(predict, data_set[0], '-bo')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()

# Root Mean Squared Error
#   RMSE = sqrt( sum( (pi – yi)^2 )/n )

error = np.subtract(predict, data_set[1])
squared_error = np.power(error, 2)
rmse = math.sqrt(np.sum(squared_error) / len(data_set[1]))
print(rmse)