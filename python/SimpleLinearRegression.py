import math
import numpy as np


class SimpleLinearRegression(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_parameters(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        x = np.subtract(self.x, x_mean)
        y = np.subtract(self.y, y_mean)
        x_y = np.multiply(x, y)
        x_pow = np.power(x, 2)
        b_1 = np.sum(x_y) / np.sum(x_pow)
        b_0 = y_mean - b_1 * x_mean

        return b_0, b_1

    def get_prediction(self):
        predict = []
        b_0, b_1 = self.get_parameters()
        for i in self.x:
            y = b_0 + (b_1 * i)
            predict.append(round(y, 2))
        return predict

    def cost_function(self):
        error = np.power((np.subtract(self.get_prediction(), self.x)), 2)
        return np.sum(error) / (2 * len(self.y))

    def root_mean_squared_error(self):
        return math.sqrt(self.cost_function())
