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


# plt.scatter(X2, X1)
# plt.show()
# plt.scatter(X3, X1)
# plt.show()

df['ones'] = 1 # bias
Y = df['X1']
# X = df[['X2', 'X3', 'ones']]

# X2only = df[['X2', 'ones']]
# X3only = df[['X3', 'ones']]


N = 30000
w = 0.1
lr = 0.000001




X= df['X2']
W = []

plt.scatter(X, Y)
plt.show()

print("X:", X)
print("Y:", Y)

for i in range(N):
    # XT = X.T
    # XW = X*w
    # XWY= XW - Y
    # XTDXWY = XT.dot(XWY)
    # step = lr * XTDXWY
    # w = w - step
    w = w - lr*X.T.dot(X*w-Y)
    # W.append(w)
    print("i:", i, " w:", w)
    # X.append(i)

Yhat = X*w

plt.scatter(X, Y,label = 'Y')
plt.plot(X, Yhat,label = 'Yhat')
plt.legend()
plt.show()


# plt.scatter(X, W,label = 'W1')
# plt.legend()
# plt.show()


# def get_r2(X, Y):
#     w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
#     Yhat = X.dot(w)
#     d1  = Y - Yhat
#     d2 = Y - Y.mean()
#     r2 = 1 - d1.dot(d1)/d2.dot(d2)
#     return r2
#
# print("r2 for X2 only:", get_r2(X2only, Y))
# print("r2 for X3 only:", get_r2(X3only, Y))
# print("r2 for X2 only:", get_r2(X, Y))