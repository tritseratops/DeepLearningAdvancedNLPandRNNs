import numpy as np
import pandas as pd

from l11_ecom_Preprocessing import get_binary

X, Y = get_binary()

D = X.shape[1]

W = np.random.randn(D)
b = 0

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classifications(Y, P):
    return np.mean(Y == P)


P_Y = forward(X, W, b)
predictions = np.round(P_Y)
# print("P_Y", P_Y)
# print("predictions", predictions)
cl = classifications(Y, predictions)
print("cl:", cl)

W = np.random.randn(D)
b = 0
P_Y = forward(X, W, b)
predictions = np.round(P_Y)
# print("P_Y", P_Y)
# print("predictions", predictions)
cl = classifications(Y, predictions)
print("cl:", cl)

W = np.random.randn(D)
b = 0
P_Y = forward(X, W, b)
predictions = np.round(P_Y)
# print("P_Y", P_Y)
# print("predictions", predictions)
cl = classifications(Y, predictions)
print("cl:", cl)


