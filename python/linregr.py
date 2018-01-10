import numpy as np
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = np.genfromtxt('../data/population_profit.txt', delimiter=',')

#Plot the data
scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
show()