import pylab as pl
import numpy as np

from pylab import *
from numpy import *
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
'''*****************************************************
     using scikit learn 
******************************************************'''
#diabetes = datasets.load_diabetes()
#diabetes_X = diabetes.data[:,np.newaxis]
#diabetes_X_temp = diabetes_X[:, :, 2]
#print diabetes_X_temp

#load dataset
dataset = loadtxt('ex1data1.txt', delimiter = ',')

x = dataset[:,0]
y = dataset[:,1]

#split the data into train and test 
x_train = x[ :-20]
y_train = y[ :-20]

m = y_train.size
it = ones(shape =(m,2))
it[:,1] = x_train


x_test = x[-20: ]
y_test = y[-20: ]
n = y_test.size
its = ones(shape =(n,2))
its[:,1] = x_test


#create a linear regression object
clf = LinearRegression()

# Train the model using the training sets
clf.fit(it, y_train)
predicted = clf.predict(it)

#print 'Coefficients: \n', regr.coef_
#np.mean((regr.predict(its) - y_test) ** 2)

# Explained variance score: 1 is perfect prediction
#regr.score(its, y_train)

pl.scatter(y_train, predicted)
#pl.plot(it[0:-1], clf.predict(it[0:-1]),color = 'blue', linewidth = 3)

pl.plot([0,30], [0,30], '--k')
pl.axis('tight')

#pl.xticks(())
#pl.yticks(())
pl.show()
