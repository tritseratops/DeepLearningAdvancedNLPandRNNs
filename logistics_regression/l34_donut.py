import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))

X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(int(N/2)) + R_outer
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
X = np.concatenate([X_inner, X_outer])

T = np.array([0]*(int(N/2)) + [1]*(int(N/2)))

plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

ones = np.array([[1]*N]).T

r  = np.zeros((N,1))
for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones, r, X), axis = 1)

w = np.random.rand(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)

# calculate cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(T.shape[0]):
        if T[i]==1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate = 0.0001
EPOCHS = 10000
l2_lambda = 0.1
ce_step = 1000

ce_log = []
for i in range(EPOCHS):
    ce = cross_entropy(T, Y)
    ce_log.append(ce)
    if i%ce_step==0:
        print("i:", i, "CE:", ce)

    # weights
    # w -= learning_rate*(T[i]*p.log(Y[i])+(1-T[i])*(1-Y[i]))) + learning_rate*(l2_lambda * w)
    w += learning_rate*((np.dot(Xb.T,(T-Y))) - l2_lambda * w)

    Y = sigmoid(Xb.dot(w))

plt.plot(ce_log)
plt.title("CE")
plt.show()


