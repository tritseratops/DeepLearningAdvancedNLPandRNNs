import numpy as np

b=np.array([[1,2,3], [7,8,9], [10,11,12], [13,14,15]])
A=np.array([[1,2,3], [4,5,6]])
print("A:", A)
print("b:",b)
print("A.T:", A.T)
print("b.T:",b.T)

def dot_print(X, Y):
    try:
        print("X.shape:", X.shape)
        print("Y.shape:", Y.shape)
        result = X.dot(Y)
        print("X.dot(Y)):", result)
        print("Result shape:", result.shape)
    except:
        print("Dot product can't be calculated")

dot_print(A, b)
dot_print(A.T, b)
dot_print(A.T, b.T)
dot_print(A, b.T)
dot_print(b, A)
dot_print(b.T, A)
dot_print(b.T, A.T)
dot_print(b, A.T)

exit()
try:
    print("A.dot(b):", A.dot(b))
except:
    print("A.dot(b) is not valid")


try:
    print("A.T.dot(b):", A.T.dot(b))
except:
    print("A.T.dot(b) is not valid")

try:
    print("b.dot(A):", b.dot(A))
except:
    print("b.dot(A) is not valid")

try:
    print("b.T.dot(A):", b.T.dot(A))
except:
    print("b.T.dot(A) is not valid")

try:
    print("A.dot(b.T):", A.dot(b.T))
except:
    print("A.dot(b.T) is not valid")

try:
    print("A.T.dot(b.T):", A.T.dot(b.T))
except:
    print("A.T.dot(b.T) is not valid")

try:
    print("b.dot(A.T):", b.dot(A.T))
except:
    print("b.dot(A.T) is not valid")

try:
    print("b.T.dot(A.T):", b.T.dot(A.T))
except:
    print("b.T.dot(A.T) is not valid")