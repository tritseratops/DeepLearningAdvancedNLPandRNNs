import numpy as np

b=np.array([1,2,3])
A=np.array([[1,2,3], [4,5,6]])
print("A:", A)
print("b:",b)
print("A.T:", A.T)
print("b.T:",b.T)
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