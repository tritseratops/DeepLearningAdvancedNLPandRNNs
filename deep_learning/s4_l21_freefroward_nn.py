import numpy as np
import matplotlib.pyplot as plt
N=500
D = 2
X1 = np.random.randn(N, D) + np.array([4,4])
X2 = np.random.randn(N, D) + np.array([0,4])
X3 = np.random.randn(N, D) + np.array([4,0])

X = np.concatenate((X1, X2, X3))

Y =  np.array([0]*500+[1]*500+[2]*500)
# plt.scatter(X[:,0], X[:, 1], c=Y)
# plt.show()

M = 3 # hidden nodes
K = 3 # output categories

# initializing weights
def init_weights(D, M, K):
    W = np.random.randn(D, M) # DxM
    b = np.random.randn(M) # bias for hidden layers
    V = np.random.randn(M, K) # MxK
    c = np.random.randn(K) # bias for output layers

    # print(W)
    # print(V)

    return W, b, V, c

def sigmoid(a):
    return 1/(1+np.exp(-a))

def forward(X, W, b, V, c):
    Z = sigmoid(X.dot(W) + b)
    Y =  np.exp(Z.dot(V) + c)
    Y_softmax =  Y/Y.sum(axis=1, keepdims=True)
    return Y_softmax

def classification_rate(predictions, targets):
    return np.mean(targets==predictions)

W, b, V, c = init_weights(D, M, K)

Yp = forward(X, W, b, V, c)
Ypr = np.argmax(Yp, axis=1)

print("Classification rate with random weights:", classification_rate(Ypr, Y))
# print(Yp.sum(axis=1))
# print(Yp)
# print(Yp.shape)
# print(Ypr.shape)
# print(Ypr)




exit()