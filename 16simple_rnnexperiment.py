# simple_rnn_test.py in sources
from __future__ import print_function, division
from builtins import range, input
# note, you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8 # sequence
D = 2 # input dimention
M = 3 # hidden dimention

X = np.random.randn(1, T, D) # we can think about it as single sentence of word vectors. 8 words by 2 feature dimentions?
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

def lstm_my():
    T = 10  # sequence - 10 word sin sentence?
    D = 100  # input dimention - 100 qualifiers of each word
    M = 20  # hidden dimention - output

    my_X = np.random.randn(1, T,
                        D)  # we can think about it as single sentence of word vectors. 8 words by 2 feature dimentions?

    input_ = Input(shape=(T,D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(my_X)
    print("o:",o)
    print("h:", h)
    print("c:", c)
    print("my_X[0][0]:", my_X[0][0])
    print("my_X.shape:", my_X.shape)

# lstm1()
# lstm2()
# gru1()
# gru2()
lstm_my()
print("X:",X)