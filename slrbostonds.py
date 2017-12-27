# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
features = boston.feature_names
target = boston.target
data = boston.data

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

X = bos['RM']
Y = bos['PRICE']
data_set = [X, Y]

slr = SimpleLinearRegression(X, Y)
b_1, b_0 = slr.get_parameters()
print("β0: %f β1: %f" % (b_0, b_1))

J = slr.cost_function()
print("Cost function: " + str(J))

predict = slr.get_prediction()

print slr.root_mean_squared_error()

plt.figure(1)
plt.subplot(211)
plt.scatter(data_set[0], data_set[1], marker='.', c='b')
plt.plot(data_set[0], predict, '-b.', color='green')
plt.title('The Boston house-price')
plt.xlabel('RM')
plt.ylabel('PRICE')
plt.grid(b=True, which='major', color='b', linestyle='-')

plt.subplot(212)
plt.scatter( predict, data_set[1],marker='.', c='r')
plt.xlabel('Prices')
plt.ylabel('Predicted prices')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.show()

