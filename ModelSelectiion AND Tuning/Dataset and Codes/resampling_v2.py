# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:03:15 2017

@author: Narayana_GLB
"""

## Cross-validation: Evaluating estimator performance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm, neighbors

## Auto data set
## Polynomial fit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
auto = pd.read_csv("Auto.csv")
X_train, X_test= train_test_split(auto, test_size = 0.4, random_state = 0)

### Linear model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 1)
coefs
print(coefs)
p = np.poly1d(coefs)
plt.plot(X_train.horsepower, X_train.mpg, "bo", markersize= 2)
plt.plot(X_train.horsepower, p(X_train.horsepower), "r-") #p(X) evaluates the polynomial at X
plt.show()
np.mean((X_train.mpg - p(X_train.horsepower))**2)

### Quandratic model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 2)
coefs
print(coefs)
p = np.poly1d(coefs)
np.mean((X_train.mpg - p(X_train.horsepower))**2)

### Cubic model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 3)
coefs
print(coefs)
p = np.poly1d(coefs)
np.mean((X_train.mpg - p(X_train.horsepower))**2)

## Different train and test split
X_train, X_test= train_test_split(auto, test_size = 0.4, random_state = 1)

### Linear model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 1)
coefs
print(coefs)
p = np.poly1d(coefs)
plt.plot(X_train.horsepower, X_train.mpg, "bo", markersize= 2)
plt.plot(X_train.horsepower, p(X_train.horsepower), "r-") #p(X) evaluates the polynomial at X
plt.show()
np.mean((X_train.mpg - p(X_train.horsepower))**2)

### Quandratic model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 2)
coefs
print(coefs)
p = np.poly1d(coefs)
np.mean((X_train.mpg - p(X_train.horsepower))**2)

### Cubic model
coefs = np.polyfit(X_train.horsepower, X_train.mpg, 3)
coefs
print(coefs)
p = np.poly1d(coefs)
np.mean((X_train.mpg - p(X_train.horsepower))**2)

### Doing linear regression with leave one out cross validation
from sklearn import cross_validation, linear_model
import numpy as np
loo = cross_validation.LeaveOneOut(len(auto.mpg))
regr = linear_model.LinearRegression()
horsepower = np.array(auto.horsepower)
mpg = np.array(auto.mpg)
#scores = cross_validation.cross_val_score(regr, horsepower, mpg, 
#                                          scoring="mean_squared_error", cv = loo, )

#print(scores.mean())

### CV error for polynomial fit 
## Check for different degrees of polynomials

### K-fold cross-validation for polynomial fit

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the Diabetes Housing dataset
columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split() # Declare the columns names
diabetes = datasets.load_diabetes() # Call the diabetes dataset from sklearn
#df = pd.DataFrame(diabetes.data) # load the dataset as a pandas data frame
df = pd.DataFrame(diabetes.data, columns=columns) # load the dataset as a pandas data frame
y = diabetes.target # define the target variable (dependent variable) as y
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print( X_test.shape)
print(y_test.shape)
# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')

from sklearn.model_selection import KFold # import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 
KFold(n_splits=2, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
 print('TRAIN:', train_index, 'TEST:', test_index)
 X_train, X_test = X[train_index], X[test_index]
 y_train, y_test = y[train_index], y[test_index]
 
from sklearn.model_selection import LeaveOneOut 
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)
   
# Necessary imports: 
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

# Perform 6-fold cross validation
#scores = cross_val_score(model, df, y, cv=6)
#print('Cross-validated scores:')
#print(scores)

# Make cross validated predictions
#predictions = cross_val_predict(model, df, y, cv=6)
#plt.scatter(y, predictions)