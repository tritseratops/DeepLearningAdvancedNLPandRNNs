import numpy as np
import matplotlib.pyplot as plt
N=500
D = 2
X1 = np.random.randn(N, D) + np.array([4,4])
X2 = np.random.randn(N, D) + np.array([0,4])
X3 = np.random.randn(N, D) + np.array([4,0])

X = np.concatenate((X1, X2, X3))

Y =  np.array([0]*500+[1]*500+[2]*500)
plt.scatter(X[:,0], X[:, 1], c=Y)
plt.show()

print(Y.shape)

exit()