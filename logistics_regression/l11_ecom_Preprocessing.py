import numpy as np
import pandas as pd

def get_data():
    data = pd.read_csv("data/10 _ecommerce_data.csv")

    data_array = data.values

    X = data_array[:, :-1]
    Y = data_array[:, -1]

    # get N, D
    N, D = X.shape


    X2 = np.zeros((N,D+3))
    # normalise is_mobile and products viewed
    X2[:, 1] = (X[:, 1]-X[:,1].mean())/X[:, 1].std()
    X2[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # one got encoding time of day
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t + D - 1] = 1

    return X2, Y


result1 = get_data()
print("result1[0].shape:", result1[0].shape)
print("result2[1].shape", result1[1].shape)
print(result1[0][0])

def get_binary():
    X, Y = get_data()
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]

    return X2, Y2

result2 =get_binary()
print("result2[0].shape", result2[0].shape)
print("result2[1].shape", result2[1].shape)
print(result2[0][0])