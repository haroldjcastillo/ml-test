import matplotlib.pyplot as plt


def plotData(x, y):
    """plots the data points and gives the figure axes labels of population and profit."""

    plt.plot(x, y, 'rx', markersize=10, label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show(block=False)  # prevents having to close the graph to move forward with ex1.py
