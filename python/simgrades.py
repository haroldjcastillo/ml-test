import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt('./data/population_profit.txt', delimiter=',')

theta = np.zeros(2)
m = data.shape[0]
X = np.c_[np.ones((m,)), data[:, 0]]
y = data[:, 1]
iterations = 1500
alpha = 0.01


def cost_function(X, y, theta):
    h = np.dot(X, theta)
    error = np.power((np.subtract(h, y)), 2)
    return np.sum(error) / (2 * m)


def gradientDescent(X, y, theta):
    J_history = np.zeros((iterations, 1))
    for iter in range(0, iterations):
        h = np.dot(X, theta) - y
        theta[0] = theta[0] - alpha * (1 / m) * np.sum(h * X[:, 0])
        theta[1] = theta[1] - alpha * (1 / m) * np.sum(h * X[:, 1])
        J_history[iter] = cost_function(X, y, theta)
    return theta, J_history


theta, j_history = gradientDescent(X, y, theta)
print("theta 0: ", theta[0], " theta 1: ", theta[1])

for i in range(X.shape[1]):
    y_predict = theta[0] + theta[1] * X

plt.figure()
plt.plot(X[:, 1], y, '.')
plt.plot(X, y_predict, 'k-')

t0 = np.linspace(-10, 10, m)
t1 = np.linspace(-1, 4, m)

jv = np.zeros((m, m))

for i in range(0, m):
    for j in range(0, m):
        t = np.array([t0[i], t1[j]])
        jv[i, j] = cost_function(X, y, t)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(t0, t1, np.transpose(jv))
ax.plot(theta[0], theta[1], 'rx')
ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
plt.show()
