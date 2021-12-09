# in original course: bilstm_list.py
from __future__ import print_function, division
from builtins import range, input


import os

import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D, Lambda, Concatenate, Dense
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def get_mnist(limit=None):
    if not os.path.exists("large_files"):
        print("You must create a folder cxalled large_files in root project directory")
    if not os.path.exists("large_files/toxic_comment_train.csv"):
        print("Looks like you haven't download data or it is not in the right spot.")
        print("Please, get train.csv from https://www.kaggle.com/c/digit-recognizer")
        print("and place in large_files folder")


    print("Reading and transforming data...")
    df = pd.read_csv("large_files/toxic_comment_train.csv")
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1,28,28)/255 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X,Y = X[:limit], Y[:limit]
    return X, Y

# get data
X, Y = get_mnist()

# config
D = 28
M = 15


# input is an image of size 28x28
input_ = Input(shape=(D,D))

# up-down
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_) # output is NxDx2M
x1 = GlobalMaxPooling1D()(x1) # output is N x 2M

