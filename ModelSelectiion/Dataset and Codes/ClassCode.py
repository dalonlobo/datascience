#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:00:13 2017

@author: dalonlobo
"""

## Cross-validation: Evaluating estimator performance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm, neighbors

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

for i in range(10):
    X_train, X_test= train_test_split(auto, test_size = 0.4, random_state = i)
    mseArray = []
    for deg in range(10):
        coefs = np.polyfit(X_train.horsepower, X_train.mpg, deg)
        coefs
        p = np.poly1d(coefs)
        mseArray.append(np.mean((X_train.mpg - p(X_train.horsepower))**2))
        
    plt.plot(mseArray)
plt.show()

