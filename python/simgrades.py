import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression

data = np.genfromtxt('../data/population_profit.txt', delimiter=',')

m = len(data[:, 0])
theta = np.zeros((2, 1))
X = np.c_[np.ones((m,)), data[:, 0]]
y = data[:, 1]
iterations = 1500
alpha = 0.001


def cost_function(X, y, theta):
    h = np.dot(X, theta)
    error = np.power((np.subtract(h, y)), 2)
    J = np.sum(error[:, 1]) / (2 * m)
    return J


J_history = np.zeros((iterations,1))

for iter in range(1, iterations):
    loss = np.subtract(np.dot(X, theta), y)
    # print("Iteration %d | Cost: %f" % (iter, cost_function(X, y, theta)))
    gradient = np.dot(X.transpose(), loss) / m
    print(gradient)
    theta = theta - alpha * gradient
    theta = theta - alpha * gradient
    J_history[iter] = cost_function(X, y, theta)

print(theta)
