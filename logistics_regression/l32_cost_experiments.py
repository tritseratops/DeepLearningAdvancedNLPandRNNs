import numpy as np
import matplotlib.pyplot as plt

def cost(T, Y, l1, w):
    one_m_t = 1 - T
    np_log_1_m_y =  np.log(1-Y)
    np_log_y =  np.log(Y)
    abs_w_mean = np.abs(w).mean()
    l_abs_w_mean  = l1 * abs_w_mean
    one_m_t_1_m_y = one_m_t*np_log_1_m_y
    t_m_y = T*np_log_y
    first_part_mean = (one_m_t_1_m_y + t_m_y).mean()
    f_cost = - first_part_mean + abs_w_mean
    return f_cost
    # return -((1-T)*np.log(1-Y) + T*np.log(Y)).mean() + l1*np.abs(w).mean()

# fat matrix
N = 51
D = 50

def sigmoid(z):
    return 1/(1+np.exp(-z))

# X = (np.random.randn(N, D) - 0.5)*10
X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))


# T = np.round(sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))
T = np.round(sigmoid(X.dot(true_w)))


Y = T
l1_lambda = 0.0001

def weight_multiplied():
    w_random = np.random.randn(D)/np.sqrt(D)
    w_x2 = true_w*2
    w_x5 = true_w*5
    w_x100 = true_w*100
    Ywx2 = sigmoid(X.dot(w_x2))
    Ywx5 = sigmoid(X.dot(w_x5))
    Y2x100 = sigmoid(X.dot(w_x100))
    Ywxrandom= sigmoid(X.dot(w_random))

    # y changed directly
    almost_zero = 0.00001
    Yx2 = T*2
    Yx2[Yx2 == 0] = almost_zero
    Yx5 = T*5
    Yx5[Yx5 == 0] = almost_zero
    Yx100 = T*100
    Yx100[Yx100 == 0] = almost_zero

    Y_inverted = np.abs(T-1)

    print(cost(T, Y, l1_lambda, true_w))
    print(cost(T,Ywxrandom, l1_lambda, true_w))
    print("Ywx2:", cost(T, Ywx2, l1_lambda, w_x2))
    print("Ywx5:", cost(T, Ywx5, l1_lambda, w_x5))
    print("Ywx5 + w_x2:", cost(T, Ywx5, l1_lambda, w_x2))
    print("Ywx100:", cost(T, Y2x100, l1_lambda, w_x100))
    print("Y_inverted:", cost(T, Y_inverted, l1_lambda, w_x100))

    print("Yx2:", cost(T, Yx2, l1_lambda, w_x2))
    print("Yx5:", cost(T, Yx5, l1_lambda, w_x5))
    print("Yx100:", cost(T, Yx100, l1_lambda, w_x100))

    plt.plot(T, label="Target")
    # plt.plot(Y_random, label="random")
    plt.plot(Ywx2, label="Tx2")
    plt.plot(Ywx5, label="Tx5")
    plt.legend()
    plt.show()

    plt.plot(T, label="Target")
    # plt.plot(Y_random, label="random")
    plt.plot(Yx2, label="Tx2")
    plt.plot(Yx5, label="Tx5")
    plt.legend()
    plt.show()

def test_diff_w():
    # 2022.06.26
    true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))
    w_05_05_m05 = np.array([0.5, 0.5, -0.5] + [0]*(D-3))
    w_0_05_m05 = np.array([0, 0.5, -0.5] + [0]*(D-3))
    w_0_0_m05 = np.array([0, 0, -0.5] + [0]*(D-3))
    w_0_0_0 = np.array([0, 0, 0] + [0]*(D-3))
    w_0_05_05_1 = np.array([0, 0.5, 0.5, 1] + [0]*(D-4))
    T = np.round(sigmoid(X.dot(true_w)))
    # we should not round Y because it will outputs infnity in cost
    Y_05_05_m05 = sigmoid(X.dot(w_05_05_m05))
    Y_0_05_m05 = sigmoid(X.dot(w_0_05_m05))
    Y_0_0_m05 = sigmoid(X.dot(w_0_0_m05))
    Y_0_0_0 = sigmoid(X.dot(w_0_0_0))
    Y_0_05_05_1 = sigmoid(X.dot(w_0_05_05_1))
    print("Y_05_05_m05:", cost(T, Y_05_05_m05, l1_lambda, w_05_05_m05))
    print("Y_0_05_m05:", cost(T, Y_0_05_m05, l1_lambda, w_0_05_m05))
    print("Y_0_0_m05:", cost(T, Y_0_0_m05, l1_lambda, w_0_0_m05))
    print("Y_0_0_0:", cost(T, Y_0_0_0, l1_lambda, w_0_0_0))
    print("Y_0_05_05_1:", cost(T, Y_0_05_05_1, l1_lambda, w_0_05_05_1))

    plot_T_Y_w(T, Y_05_05_m05, "Y_05_05_m05", true_w,  w_05_05_m05, "w_05_05_m05")
    plot_T_Y_w(T, Y_0_05_m05, "Y_0_05_m05", true_w, w_0_05_m05, "w_0_05_m05")
    plot_T_Y_w(T, Y_0_0_m05, "Y_0_0_m05", true_w, w_0_0_m05, "w_0_0_m05")
    plot_T_Y_w(T, Y_0_0_0, "Y_0_0_0", true_w, w_0_0_0, "w_0_0_0")
    plot_T_Y_w(T, Y_0_05_05_1, "Y_0_05_05_1", true_w, w_0_05_05_1, "w_0_05_05_1")

    # plt.plot(true_w, label="true_w")
    # plt.plot(w_05_05_m05, label="w_05_05_m05")
    # plt.legend()
    # plt.show()
    # plt.plot(T, label="T")
    # plt.plot(Y_05_05_m05, label="Y_05_05_m05")
    # plt.legend()
    # plt.show()



    # plt.plot(w_0_05_m05, label="w_0_05_m05")
    # plt.plot(w_0_0_m05, label="w_0_0_m05")
    # plt.plot(w_0_0_0, label="w_0_0_0")
    # plt.plot(w_0_05_05_1, label="w_0_05_05_1")
    # plt.legend()
    # plt.show()
    #
    #
    # plt.plot(T, label="T")
    # plt.plot(Y_05_05_m05, label="Y_05_05_m05")
    # plt.plot(Y_0_05_m05, label="Y_0_05_m05")
    # plt.plot(Y_0_0_m05, label="Y_0_0_m05")
    # plt.plot(Y_0_0_0, label="Y_0_0_0")
    # plt.plot(Y_0_05_05_1, label="Y_0_05_05_1")
    # plt.legend()
    # plt.show()

def plot_T_Y_w(T, Y, Y_label,  T_w, Y_w, Y_w_label):
    plt.plot(T_w, label="true_w")
    plt.plot(Y_w, label=Y_w_label)
    plt.legend()
    plt.show()
    plt.plot(T, label="T")
    plt.plot(Y, label=Y_label)
    plt.legend()
    plt.show()


test_diff_w()