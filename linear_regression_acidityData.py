''' *********************************************************
This code is to analyse linear Regression and Gradient Decent
***********************************************************'''

import numpy as np 
import pylab as pl
import csv

from pylab import *
from numpy import *

def compute_cost(x, y, theta):
#number of training examples
	m = y.size

	predictions = x.dot(theta).flatten()

	sqError = (predictions - y.astype(np.float))**2

	J = (1.0/(2.0 * m)) *sqError.sum()

	return J

def gradient_descent(x, y, theta, alpha, iterations):
## numer of training samples
    m = y.size
    J_history = zeros(shape = (iterations, 1))

    for i in xrange(iterations):
        predictions = x.dot(theta).flatten()
        error_x0 = (predictions - y.astype(np.float))*x[:,0]
        error_x1 = (predictions - y.astype(np.float))*x[:,1]
	theta[0][0] = theta[0][0] - (alpha*(1.0 / m) * error_x0.sum())
	theta[1][0] = theta[1][0] - alpha*(1.0 / m) * error_x1.sum()
    	J_history[i, 0] = compute_cost(x, y, theta)
    return theta, J_history



''' **************************************************
begin main code 
******************************************************
'''
dataset = [ ]
with open ('dataset.csv', 'rb') as csvfile:
    csv_read = csv.reader(csvfile, delimiter = ";")
    header = csv_read.next()
    for row in csv_read:
        dataset.append(row)


dataset = np.array(dataset)

'''
the dataset contains the following features. For the purpose of this we will choose fixed acidity and density
"fixed acidity";"volatile acidity";"citric acid";
"residual sugar";"chlorides";"free sulfur dioxide";
"total sulfur dioxide";"density";"pH";
"sulphates";"alcohol";"quality"
'''

#assign data to varaibles x and y
x = dataset[:,8]
y = dataset[:,0]

####a = y[:,np.newaxis]


##plot data set
#pl.scatter(x, y)
#pl.axis('tight')
#pl.show()


#declare varaibles
#number of observations 
m = y.size


# add a column to the x:
it = ones(shape = (m, 2))
it[:,1] = x
 

#declare theta
theta = zeros(shape = (2,1))


#some gradient descent settings
iterations = 10000
alpha = 0.01

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print 'value of theta', theta

 
##Plot the results
result = it.dot(theta).flatten()
pl.scatter (dataset[:,0], result, color = "red")
pl.show()
 
 
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

##Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
pl.contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
pl.xlabel('theta_0')
pl.ylabel('theta_1')
pl.scatter(theta[0][0], theta[1][0])
pl.show()




