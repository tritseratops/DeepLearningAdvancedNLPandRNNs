import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]
              ])

T = np.array([0,1,1,0])

ones = np.ones((N,1))

xy = (X[:,0]*X[:, 1]).reshape(N, 1)
Xb = np.concatenate((ones, xy, X), axis = 1)

w = np.random.randn(D + 2)

z  = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0
    for i in range(T.shape[0]):
        if T[i]==1:
            E -=np.log(Y[i])
        else:
            E -=np.log(1 - Y[i])
    return E

learning_rate = 0.01
ce_log = []
EPOCHS = 100
lambda_l1 = 0.01

for i in range(EPOCHS):
    ce = cross_entropy(T, Y)
    ce_log.append(ce)

    w += learning_rate * (Xb.dot(T-Y) + lambda_l1*w)
    Y = sigmoid(Xb.dot(w))

    print("i:", i, " CE:", ce)

plt.plot(ce_log)
plt.title("Cross entropy")
plt.show()

print("Final w:", w)
print("Final Classification rate:", 1- np.abs(T - np.round(Y)).sum()/N)

