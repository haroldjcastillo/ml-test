import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
features = boston.feature_names
target = boston.target
data = boston.data

bos = pd.DataFrame(boston.data)
bos.columns = boston.feature_names
bos['PRICE'] = boston.target

print(boston.DESCR)

# x = bos['RM']
# y = bos['ZN']

Y = bos['PRICE']
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z)
plt.show()
