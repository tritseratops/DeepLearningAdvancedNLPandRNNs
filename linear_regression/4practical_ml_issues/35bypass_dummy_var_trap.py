import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N,D))
X[:,0] = 1
X[:5,1] = 1
X[-5:,2] = 1


print("X:", X)


Y = np.array([0]*5 + [1]*5)

# w = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) # does not work or this matrixes

lr = 0.0001
epochs = 10000
w = np.random.randn(D)/np.sqrt(D) # ensure it has variance one of the D
costs = []

for i in range(epochs):
    XT= X
    Yhat_temp = XT.dot(w)
    errors= Yhat_temp - Y
    mse = errors.dot(errors)/N
    costs.append(mse)
    gradient = XT.T.dot(errors)
    step = lr * gradient
    w = w - step
    # w = w - lr*X.T.dot(X*w-Y) # full formula
    print("i:", i, " w:", w, " Step: ", step)
    # X.append(i)

Yhat = X.dot(w)

def get_r2(X, Y, Yhat):
    d1  = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r2


print("X:", X)
print("Yhat:", Yhat)
print("Gradient w:", w)
print("Gradient R2:", get_r2(X, Y, Yhat))
plt.plot(costs)
plt.show()
plt.plot(Y,label = 'data')
plt.plot(Yhat,label = 'GD prediction')
plt.legend()
plt.show()