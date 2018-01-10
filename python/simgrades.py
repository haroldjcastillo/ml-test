import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression

data = np.genfromtxt('../data/population_profit.txt', delimiter=',')

theta = np.zeros(2)
m = data.shape[0]
X = np.c_[np.ones((m,)), data[:, 0]]
y = data[:, 1]
iterations = 1500
alpha = 0.001


def cost_function(X, y, theta):
    h = np.dot(X, theta)
    error = np.power((np.subtract(h, y)), 2)
    J = np.sum(error[:, 1]) / (2 * m)
    return J


J_history = np.zeros((iterations, 1))

for iter in range(1, iterations):
    h = np.transpose(np.subtract(np.dot(X, theta), y))

    if iter == 3:
        print(X.shape)
        print(np.mat(y).shape)
        print(theta.shape)
        print(h.shape)
        print(X * theta)
        break
    # print()
    # # print("Iteration %d | Cost: %f" % (iter, cost_function(X, y, theta)))
    # gradient = np.dot(X.transpose(), loss) / m
    # # print(gradient)
    # theta = theta - alpha * gradient
    # theta = theta - alpha * gradient
    # J_history[iter] = cost_function(X, y, theta)

print(theta)


def gradient_descent_2(alpha, x, y, numIterations):
    # number of samples
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        print "iter %s | J: %.3f" % (iter, J)
        gradient = np.dot(x_transpose, loss) / m
        theta = theta - alpha * gradient  # update
    return theta


theta = gradient_descent_2(alpha, X, y, iterations)
print theta

for i in range(X.shape[1]):
    y_predict = theta[0] + theta[1]*X
plt.plot(X[:,1],y,'o')
plt.plot(X,y_predict,'k-')
plt.show()
print "Done!"
