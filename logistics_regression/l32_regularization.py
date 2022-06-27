import numpy as np
import matplotlib.pyplot as plt

# fat matrix
N = 51
D = 50

def sigmoid(z):
    return 1/(1+np.exp(-z))

# X = (np.random.randn(N, D) - 0.5)*10
X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

T = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))


# let's plot the data to see what it looks like
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=T)
plt.show()

# init
w = np.random.randn(D)/np.sqrt(D)
learning_rate = 0.001
l1_lambda = 0.0001
EPOCHS = 10000

def cost(T, Y, l1, w):
    return -((1-T)*np.log(1-Y) + T*np.log(Y)).mean() + l1*np.abs(w).mean()

cost_logs = []
for i in range(EPOCHS):
    Yhat = sigmoid(X.dot(w))
    delta = Yhat -T
    w -= learning_rate * (X.T.dot(delta) + l1_lambda * np.sign(w))
    ce = cost(T, Yhat, l1_lambda, w)
    cost_logs.append(ce)

    if i%1000 == 0:
        print(i, ce)

plt.plot(cost_logs)
plt.show()

plt.plot(true_w, label = "True W")
plt.plot(w, label = "W")
plt.legend()
plt.show()

print("True w:", true_w)
print("W:", w)
# plt.plot(X)
# plt.scatter(X, Y)
plt.show()



