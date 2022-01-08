import numpy as np
import keras.backend as K
from keras.layers import Dot

val = [[[0, 1],
  [2, 4]]]
# val = np.random.randint(2, size=(1, 2, 3))
a = K.variable(value=val)
val2 = [[[3, 7],
  [10,100]]]
# val2 = np.random.randint(2, size=(1, 2, 3))
b = K.variable(value=val2)
print("a")
print(val)
print("b")
print(val2)
out = Dot(axes = 2)([a,b])
print(out.shape)
print("DOT")
print(out)
print(K.eval(out))