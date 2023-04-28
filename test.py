import numpy as np
from numba import cuda

@cuda.jit
def kernel(d_out, d_x0, d_y0):
    i, k = cuda.grid(2)

    if i < d_x0.shape[0] and k < d_y0.shape[0]:
        print(i,k,d_x0[i],d_y0[k])
        d_out[i, k] = d_x0[i] + d_y0[k]


# Create input arrays
d_x0 = cuda.to_device(np.array(range(10)))
d_y0 = cuda.to_device(np.array(range(20)))

# Create output array
d_out = cuda.device_array((10, 20))

# Define block and grid sizes
threads_per_block = (16, 16)
blocks_per_grid_x = (d_x0.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (d_y0.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch kernel
kernel[blocks_per_grid, threads_per_block](d_out, d_x0, d_y0)

# Copy output back to host
out = d_out.copy_to_host()

print(out)
