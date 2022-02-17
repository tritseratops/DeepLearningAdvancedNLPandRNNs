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

X2only = df[['X2', 'ones']]
# X3only = df[['X3', 'ones']]


N = 3000
w = np.array([58, 1.46])
lr = 0.00001




# adding bias
X = np.vstack([np.ones(X2.shape[0]), X2]).T

# X= df['X2']
# X = X2only
W = []

plt.scatter(X2, Y)
plt.show()

print("X:", X)
print("Y:", Y)

num_features = X.shape[1]
w = np.zeros(num_features) # 2

for i in range(N):
    # XT = X.T
    XT= X  # XT [11, 2]
    Yhat_temp = XT.dot(w)  # Yhat_temp [11,]
    errors= Yhat_temp - Y # errors - [11,]
    gradient = XT.T.dot(errors)
    step = lr * gradient
    w = w - step
    # w = w - lr*X.T.dot(X*w-Y)
    # W.append(w)
    print("i:", i, " w:", w, " Step: ", step)
    # X.append(i)

Yhat = X.dot(w)

w_l2r = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat_l2r = X.dot(w_l2r)


def get_r2(X, Y, Yhat):
    d1  = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2


print("X:", X)
print("Yhat:", Yhat)
print("Gradient w:", w)
print("Gradient R2:", get_r2(X, Y, Yhat))
print("L2R w:", w_l2r)
print("L2R R2:", get_r2(X, Y, Yhat_l2r))
plt.scatter(X2, Y,label = 'Y')
plt.plot(X2, Yhat,label = 'Yhat')
plt.plot(X2, Yhat_l2r,label = 'L2R')
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