import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)
X[:50, :] = X[:50, :] + 2*np.ones((50, D))
X[50:, :] = X[50:, :] + -2*np.ones((50, D))
ones = np.ones((N,1))
# print("ones:",ones)
# print("X:",X)
Xb = np.concatenate((ones, X), axis=1)

T = np.ones(N)
T[50:]=0

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cross_entropy_error(Yh, Y):
    E = 0
    for i in range(Yh.shape[0]):
        if Y[i]==1:
            E -= np.log(Yh[i])
        else:
            E -=np.log(1-Yh[i])
    return E

# init random weights
w = np.random.randn(D+1)

# prediction
z = Xb.dot(w)

# squeezing to  near 0..1
Yh = sigmoid(z)

E = cross_entropy_error(Yh, T)

print("w:", w, "\nE:", E)

w = np.array([0, 4 , 4])

# prediction
z = Xb.dot(w)

# squeezing to  near 0..1
Yh = sigmoid(z)

E = cross_entropy_error(Yh, T)

print("w:", w, "\nE:", E)

