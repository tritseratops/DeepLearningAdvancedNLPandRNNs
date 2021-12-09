# simple_rnn_test.py in sources
from __future__ import print_function, division
from builtins import range, input
# note, you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)
# print(X)


def lstm1():
    input_ = Input(shape=(T,D))
    rnn = LSTM(M, return_state = True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o,h,c = model.predict(X)
    print("o:",o)
    print("h:", h)
    print("c:", c)

def lstm2():
    input_ = Input(shape=(T,D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print("o:",o)
    print("h:", h)
    print("c:", c)

def gru1():
    input_  = Input(shape=(T,D))
    rnn = GRU(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print("o:",o)
    print("h:", h)

def gru2():
    input_ = Input(shape=(T,D))
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print("o:", o)
    print("h:", h)

lstm1()
lstm2()
gru1()
gru2()
print(X)