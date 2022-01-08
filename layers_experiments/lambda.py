import keras.backend as K
from keras.layers import Dot, Lambda

val = [[[0, 1],
  [2, 4]]]
# val = np.random.randint(2, size=(1, 2, 3))
a = K.variable(value=val)
val2 = [[[3, 7 ],
  [10,100]]]
val3 = [[
  [3, 7 ,11],
  [10,100,1000],
        [5,25,125]]]
c = K.variable(value=val3)
val4 = [[
  [3, 9 ],
  [10,100]],
        [[5,25],
         [7,11]]]
d = K.variable(value=val4)
# val2 = np.random.randint(2, size=(1, 2, 3))
# b = K.variable(value=val2)
print("a")
print(a)
# print("b")
# print(val2)
# out = Dot(axes = 2)([a,b])
out = Lambda(lambda x: K.sum(x, axis=2))(a)
print(out.shape)
print("LAMBDA")
print(out)
print(K.eval(out))