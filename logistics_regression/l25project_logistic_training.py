import numpy as np
import matplotlib.pyplot as plt
from l11_ecom_Preprocessing import get_binary
from sklearn.utils import shuffle

X, Y = get_binary()

X, Y = shuffle(X, Y)

Xtest = X[:100]
Ytest = Y[:100]
Xtrain = X[100:]
Ytrain = Y[100:]

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(X, w, b):
    return sigmoid(X.dot(w)+b)

def cross_entropy(T, Py):
    return -np.mean(T*np.log(Py)+ (1-T)*np.log(1-Py))

def classification_rate(Y, Py):
    return np.mean(Y == Py)

# init
learning_rate = 0.001

D = X.shape[1]

w = np.random.randn(D)

b = 0

epochs = 10000

train_ce_data=[]
test_ce_data=[]
for i in range(epochs):
      # get cross entropy, add to data

    PyTest = forward(Xtest, w , b)
    PyTrain = forward(Xtrain, w, b)
    train_ce = cross_entropy(Ytrain, PyTrain)
    test_ce = cross_entropy(Ytest, PyTest)
    train_ce_data.append(train_ce)
    test_ce_data.append(test_ce)

    # iterating by graddient descent
    w -= learning_rate * Xtrain.T.dot(PyTrain - Ytrain)
    b -= learning_rate * (PyTrain - Ytrain).sum()
    if i%1000==0:
        print (i, train_ce, test_ce)

x_axis = np.arange(epochs)
plt.plot(x_axis, train_ce_data, label='training costs')
plt.plot(x_axis, test_ce_data, label='test costs')
plt.legend()
plt.show()








