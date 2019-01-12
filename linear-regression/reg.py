#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 13:32:51 2019

@author: aiktc
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('TKAgg') for MAC
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,:-1].values

#splitting the dataset into the training and test sets
from sklearn.model_selection import train_test_split #used model_selection in place of cross_validation since the latter is deprecated 

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 1/3,random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#visualising the training set result
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience(train)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience(test)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()