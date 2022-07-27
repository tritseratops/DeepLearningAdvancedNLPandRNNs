import numpy as np

z  = np.random.randn(100,5)

zexp = np.exp(z)
z_sf = zexp/zexp.sum(axis=1, keepdims=True)

z_sf_sum = z_sf.sum(axis=1)

print(z_sf_sum)

z_sum = zexp.sum(axis=1)

print(z_sum)

exit()