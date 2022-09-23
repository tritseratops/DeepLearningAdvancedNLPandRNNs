import numpy as np
import pandas as pd

"""

"""
def load_data():
    data_file = pd.read_csv("ecommerce_data.csv")

    data = data_file.values

    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    # is_mobile - is binary
    # n_products_viewed - should be normalized
    # visit_duration - to normalize
    # is_returning_visitor - binary
    # time_of_day - categories should be hot encoded
    # user_action - 4 action - filter to 2

    N, D = X.shape
    X2 = np.zeros((N, D+3)) # +3 because of hot encoding last column
    X2[:, 0]=X[:, 0] # is_mobile - is binary
    X2[:, 1]=(X[:, 1]-X[:, 1].mean())/X[:,1].std() # n_products_viewed - should be normalized
    X2[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std() # visit_duration - to normalize
    X2[:, 3] = X[:, 3] # is_returning_visitor - binary



    # one-hot encode time of day, 0-3
    for i in range(N):
        X2[i, D+int(X[i, 4])-1] = 1

    print(X2.shape)
    print(X2)
    return X2, Y


# load_data()