import numpy as np
import matplotlib.pyplot as plt

N = 30
w = 20
lr = 0.1

# for i in range(N):
#     w = w - lr*2*w
#     print("i:", i, " w:", w)

N = 30
w1 = 0.75
w2 = 0.75
lr = 0.445


X= []
W1 = []
W2 = []


for i in range(N):
    w1 = w1 - lr*2*w1
    W1.append(w1)
    print("i:", i, " w1:", w1)
    w2 = w2 - lr*4*(w2**3)
    W2.append(w2)
    print("i:", i, " w2:", w2)
    X.append(i)

plt.scatter(X, W1,label = 'W1')
plt.scatter(X, W2, label = 'W2')
plt.legend()
plt.show()

N = 5
w1 = 20
w2 = 20
lr = 0.1


X= []
W1 = []
W2 = []


for i in range(N):
    w1 = w1 - lr*2*w1
    W1.append(w1)
    print("i:", i, " w1:", w1)
    w2 = w2 - lr*4*(w2**3)
    W2.append(w2)
    print("i:", i, " w2:", w2)
    X.append(i)

plt.scatter(X, W1,label = 'W1')
plt.scatter(X, W2, label = 'W2')
plt.legend()
plt.show()