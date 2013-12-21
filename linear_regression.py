''' *********************************************************
This code is to analyse linear Regression and Gradient Decent
***********************************************************'''

import numpy as np 
import pylab
import csv

from pylab import *
from numpy import *

def compute_cost(x, y, theta):
#number of training examples
	m = y.size

	predictions = x.dot(theta).flatten()

	sqError = (predictions - y)**2

	J = (1.0/(2.0 * m)) *sqError.sum()

	return J

def gradient_decent(x, y, theta, alpha, iterations):
# numer of training samples
    m = y.size

    J_history = zeros(shape = (iterations, 1))
    
    for i in xrange(iterations):
    	predictions = x.dot(theta).flatten()
    	error_x0 = (predictions - y)*x[:,0]
    	error_x1 = (predictions - y)*x[:,1]

    	theta[0][0] = theta[0][0] - alpha*(1.0 / m) * error_x0.sum()
    	theta[1][0] = theta[1][0] - alpha*(1.0 / m) * error_x1.sum()

    	J_history[i, 0] = compute_cost(x, y, theta)

    return theta, J_history


#load dataset as a text file
dataset = loadtxt('ex1data1.txt', delimiter = ',')


#assign data to varaibles x and y
x = dataset[:,0]
y = dataset[:,1]

#declare varaibles

#number of observations 
m = y.size

# add a column to the x:
it = ones(shape = (m, 2))
it[:,1] = x

#declare theta
theta = zeros(shape = (2,1))


#some gradient descent settings
iterations = 1500
alpha = 0.01

gradient_decent(it, y, theta, alpha, iterations)






print theta
#Predict values for population sizes of 35,000 and 70,000
predict1 = array([1, 3.5]).dot(theta).flatten()
print 'For population = 35,000, we predict a profit of %f' % (predict1 * 10000)
predict2 = array([1, 7.0]).dot(theta).flatten()
print 'For population = 70,000, we predict a profit of %f' % (predict2 * 10000)
 
#Plot the results
result = it.dot(theta).flatten()
plot(dataset[:, 0], result)
show()
 
 
#Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)
 
 
#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))
 
#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(it, y, thetaT)
 
#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()




