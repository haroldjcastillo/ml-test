# coding=utf-8
import math
import numpy as np
from matplotlib import pyplot as plt

# Simple Linear Regression

#   y = B0 + B1 * x
#   B1 = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#   B0 = mean(y) – B1 * mean(x)

# data_set = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 4, 5, 6, 8, 9, 10, 13, 12]]
data_set = [[1, 2, 4, 3, 5], [1, 3, 3, 2, 5]]

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

# Cost function J
# 1/m * ∑(ÿ - (β0 + β1*Xi))^2
# J(β0, β1)
error = np.subtract(predict, data_set[1])
squared_error = np.power(error, 2)
J = np.sum(squared_error) / len(data_set[1])
print("Cost function: " + str(J))

# Root Mean Squared Error
#   RMSE = sqrt( sum( (pi – yi)^2 )/n )
rmse = math.sqrt(J)
print("Root Mean Squared Error: " + str(rmse))

plt.scatter(data_set[0], data_set[1], marker='o', c='b')
plt.plot(predict, data_set[0], '-bo', color='green')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.text(3, 8, 'boxed italics text in data coords')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()
