import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    # function:  y=x1*x2

    N = 1000
    D = 2

    X = np.random.randn(N, D)
    X = X*2/max(abs(X.min()),X.max()) # make range from -2 to 2

    Y = X[:, 0]*X[:, 1]

    print(X)
    print(X.max())
    print(X.min())
    print(X.mean())
    print(Y)

    plot_data(X, Y, "Target", 'r')

    return X, Y

def plot_data(X, Y, legend, color):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.grid()

    ax.scatter(X[:, 0], X[:, 1], Y, c=color, s=50)
    ax.set_title('Y=X1*X2')

    # Set axes label
    ax.set_title(legend)
    ax.set_xlabel('X1', labelpad=20, fontsize=18)
    ax.set_ylabel('X2', labelpad=20, fontsize=18)
    ax.set_zlabel('Y', labelpad=20, fontsize=18)
    plt.legend()
    plt.show()
# generate_data()