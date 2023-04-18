import numpy as np
from numba import cuda
gpu = cuda.get_current_device()
x = np.array([0])
d_x = cuda.to_device(x)
@cuda.jit
def kernel(x):
    n = cuda.grid(1)
    x[n] = 1
kernel[1,1](d_x)
x = d_x.copy_to_host()
print(x)
