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

def save_3d_array(my_array):
    # stacked = pd.Panel(my_array.swapaxes(1, 2)).to_frame().stack().reset_index()
    stacked = np.reshape(my_array, (1, np.product(my_array.shape)))
    print(len(stacked))
    stacked.columns = ['x', 'y', 'z', 'value']
    # save to disk
    stacked.to_csv('stacked.csv', index=False)


def get_mnist(limit=None):
    if not os.path.exists("large_files"):
        print("You must create a folder cxalled large_files in root project directory")
    if not os.path.exists("large_files/train.csv"):
        print("Looks like you haven't download data or it is not in the right spot.")
        print("Please, get train.csv from https://www.kaggle.com/c/digit-recognizer")
        print("and place in large_files folder")


    print("Reading and transforming data...")
    df = pd.read_csv("large_files/train.csv")
    data = df.values
    print("type data:",type(data))
    print("data:", data)
    print("data shape:", data.shape)
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1,28,28) / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X,Y = X[:limit], Y[:limit]
    # save_3d_array(Y)
    np.savetxt("Y.csv", Y, delimiter=",")
    # pd.DataFrame(X).to_csv("X.csv")
    # plt.imshow(X[0], interpolation='nearest')
    # plt.show()
    # plt.imshow(X[1], interpolation='nearest')
    # plt.show()
    # plt.imshow(X[2], interpolation='nearest')
    # plt.show()
    # plt.imshow(X[3], interpolation='nearest')
    # plt.show()
    return X, Y

# get data
X, Y = get_mnist()
print("X:",X)
print("X type",type(X))
print("X shape",X.shape)
print("Y:",Y)
print("Y type",type(Y))
print("Y shape",Y.shape)

# config
D = 28
M = 15


# input is an image of size 28x28
input_ = Input(shape=(D,D))

# up-down
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_) # output is NxDx2M
x1 = GlobalMaxPooling1D()(x1) # output is N x 2M



# custom layer
permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))
x2 = permutor(input_)

# left-right
rnn2 = Bidirectional(LSTM(M, return_sequences=True))
x2 = rnn2(x2) # output is N x D x 2M
x2 = GlobalMaxPooling1D()(x2)

print("x2.shape:", x2.shape)

# put them together
concatenator = Concatenate(axis=1)
x  = concatenator([x1,x2]) # output is N x 4M
print("x.shape:", x.shape)
print("x:", x)


# final dense layer
output = Dense(10, activation='softmax')(x)

model = Model(inputs= input_, outputs = output)

# testing
o = model.predict(X)
print("o.shape:", o.shape)

# compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

# train
print('Training model...')
r = model.fit(X,Y, batch_size=32, epochs=10, validation_split=0.3)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()