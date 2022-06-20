import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from l11_ecom_Preprocessing import get_binary

X, Y = get_binary()

X, Y  = shuffle(X, Y)

Xtrain = X[:-100, :]
Ytrain = Y[:-100]
Xtest = X[-100:, :]
Ytest = Y[-100:]

def sigma(z):
    return 1/(1+np.exp(-z))

def cross_entropy(T, Y):
    return - np.mean(T*np.log(Y)+(1-T)*np.log(1-Y))

def forward(X, w, b):
    return sigma(X.dot(w)+b)

def classification_rate(T, Py):
    return np.mean(T==Py)

D = X.shape[1]
EPOCHS = 10000
learning_rate = 0.001
w = np.random.randn(D)
b = 0

train_ce_log = []
test_ce_log = []

for i in range(EPOCHS):
    # get predictions
    PyTrain = forward(Xtrain, w , b)
    PyTest =  forward(Xtest, w , b)

    # calc cross entropy
    train_ce = cross_entropy(Ytrain, PyTrain)
    test_ce = cross_entropy(Ytest, PyTest)

    train_ce_log.append(train_ce)
    test_ce_log.append(test_ce)

    # calc new w nd b
    w -= learning_rate* (Xtrain.T.dot(PyTrain-Ytrain) -learning_rate * w)
    b -= learning_rate* ((PyTrain-Ytrain).sum())

    if i%1000==0:
        print("Training CE:", train_ce, " Test CE:", test_ce)


# plot
x_axis = np.arange(EPOCHS)
plt.plot(x_axis, train_ce_log, label = "train")
plt.plot(x_axis, test_ce_log, label = "test")
plt.legend()
plt.show()
print("W:", w, " b:", b)
print("Final Classification rate train:", classification_rate(Ytrain, np.round(PyTrain)))
print("Final Classification rate test:", classification_rate(Ytest, np.round(PyTest)))