import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """Plot the positive and negative examples on a 2D plot, using the option 'k+' for the positive examples and 'ko'
    for the negative examples."""
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    p1 = plt.plot(X[pos, 0], X[pos, 1], marker='+', markersize=9, color='k')[0]
    p2 = plt.plot(X[neg, 0], X[neg, 1], marker='o', markersize=7, color='y')[0]

    return plt, p1, p2