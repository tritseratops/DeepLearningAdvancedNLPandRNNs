import numpy as np

# initializing input data
N=100
D = 2

X = np.random.randn(N,D)

# first 50 center -2  second 50 center +2
X[:50, :] = X[:50, :] - 2*np.ones((50,D))
X[50:, :] = X[50:, :] + 2* np.ones((50,D))

Y = np.zeros(N).T
Y[:50] = 1

# bias?
ones = np.ones((N,1))
# ones = np.array([1]*N).T

Xb= np.concatenate((ones, X), axis=1)



# get sigmoid
def sigmoid(z):
    return 1/(1+np.exp(z))

# calculate cross entropy error
def cross_extropy_error(T, Y):
    E = 0
    items = T.shape[0]
    for n in range(items):
        if T[n]==1:
            E -=np.log(Y[n])
        else:
            E -= np.log(1-Y[n])
    return E

# init random weights
w = np.random.randn(D)

w = np.concatenate((w, np.ones(1)))

# get prediction
Yh = Xb.dot(w)

Xs = sigmoid(Y)

print(cross_extropy_error(Y, Yh))

w = np.array([0, 4, 4])
Yh = Xb.dot(w)
Xs = sigmoid(Y)
print(cross_extropy_error(Y, Yh))