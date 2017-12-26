from numpy import loadtxt, ones, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = loadtxt('ex1data1.txt', delimiter=',')

#Plot the data
scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
show()