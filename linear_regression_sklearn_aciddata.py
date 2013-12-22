import pylab as pl
import numpy as np
import csv
from pylab import *
from numpy import *
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

''' *****************************************
using scikit learn 
******************************************'''
dataset = []
#read dataset found here: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
with open('ex1data1.csv', 'rb') as csvfile:
    csv_read = csv.reader(csvfile, delimiter = ",")
    header = csv_read.next()
    for row in csv_read:
        dataset.append(row)
dataset = np.array(dataset)

#declar independent and dependent variables
x = dataset[:,0].astype(np.float)
y = dataset[:,1].astype(np.float)

#split the data into train and test 
x_train = x[:-20]
y_train = y[:-20]

#number of datasets
m = x_train.size
it = ones(shape = (m,2))
it[:,1] = x_train

#decalre test data
#x_test = x[-100:]
#y_test = y[-100:]
#n = y_test.size
#its = ones(shape =(n,2))
#its = [:,1] = x_test

#create a linear regression object 
clf = LinearRegression()

#train the model using the train sets
clf.fit(it, y_train)
predicted = clf.predict(it)

print 'Coefficients:\n', clf.coef_
np.mean((clf.predict(it) - y_train)**2)
print clf.score(it, y_train)
print clf.intercept_

pl.scatter(y_train, predicted)
pl.plot([0,30],[0,30], '--k')
pl.axis('tight')

pl.show()

#Train the model 
#clf.fit(it, y_train)
#predicted = clf.predict(it)
#pl.scatter(y_train, predicted)
#pl.plot([0,5],[0,5],'--k')
#pl.axis('tight')

#pl.xticks(())
#pl.yticks(())
#pl.show()
