# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# need pip intall openpyxl
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xlrd

X1=X2=X3 = None
df = pd.read_excel("mlr02.xls", engine='xlrd')
X = df.values
X1 = df['X1'] # sistolic pressure
X2 = df['X2'] # age
X3 = df['X3'] # weight
min = df.min().values.min()
max = df.max().values.max()
# print("df.min().values.min()", df.min().values.min())
print("min:", min)
print("max:", max)
df['X4']= np.random.randint(min, max, df.shape[0]) # noise

print("X1.shape:",X1.shape)
print("df:",df)
exit()
plt.scatter(X2, X1)
plt.show()
plt.scatter(X3, X1)
plt.show()

df['ones'] = 1 # bias
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = X.dot(w)
    d1  = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2

print("r2 for X2 only:", get_r2(X2only, Y))
print("r2 for X3 only:", get_r2(X3only, Y))
print("r2 for X2 only:", get_r2(X, Y))