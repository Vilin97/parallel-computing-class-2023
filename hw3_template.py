import numpy as np
import matplotlib.pyplot as plt
# import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "1"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "1";
from numba import cuda, jit
import math
from time import time

PI = np.pi

cc_cores_per_SM_dict = {
	(2,0) : 32,
	(2,1) : 48,
	(3,0) : 192,
	(3,5) : 192,
	(3,7) : 192,
	(5,0) : 128,
	(5,2) : 128,
	(6,0) : 64,
	(6,1) : 128,
	(7,0) : 64,
	(7,5) : 64,
	(8,0) : 64,
	(8,6) : 128,
	(8,9) : 128,
	(9,0) : 128
	}

def print_device_info():
	'''
	Query device properties and print name of GPU model, compute capability, 
	number of SMs, number of cores per SM
	'''
	device = cuda.get_current_device()
	compute_capability = device.compute_capability
	name = device.name
	num_SMs = device.MULTIPROCESSOR_COUNT
	cores_per_SM = cc_cores_per_SM_dict[compute_capability]
	max_threads_per_block = device.MAX_THREADS_PER_BLOCK
	print("Device name: ", name)
	print("Compute capability: ", compute_capability)
	print("Number of SMs: ", num_SMs)
	print("Number of cores per SM: ", cores_per_SM)
	print("Max threads per block: ", max_threads_per_block)

def p1():
	print("\nProblem 1")
	print_device_info()

@jit	
def f(x,a):
	'''
	Compute the output of the map x->f(x,a) = a(sin(PI*x))
	Args:
		x: float input value
		a: float parameter value
	Returns:
		float output value
	'''
	return a*math.sin(PI*x)

@jit
def iterate_f(x0,a,m):
	'''
	Compute the m^th iterate of the map x->f(x,a) = a(sin(PI*x))
	Args:
		x0: float initial value
		a: float parameter value
		m: int iteration count
	Returns:
		float value of m^th iterate
	'''
	x = x0
	for i in range(m):
		x = f(x, a)
	return x

def array_iterate_f(v,a,m):
	'''
	Compute an array of m^th iterates of the map x->f(x,a) = a(sin(PI*x))
	Args:
		v: 1D float numpy array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		1D float numpy array of m^th iterates
	'''
	n = v.shape[0]
	result = np.zeros_like(v)
	for i in range(n):
		result[i] = iterate_f(v[i], a, m)
	return result
		
def timed_array_iterate_f(v,a,m):
	'''
	Compute an array of m^th iterates of the map x->f(x,a) = a(sin(PI*x))
	Args:
		v: 1D float numpy array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		result: 1D float numpy array of m^th iterates
		elapsed: float computation time in ms
	'''
	start = time()
	result = array_iterate_f(v,a,m)
	elapsed = 1000*(time()-start)
	return result, elapsed
	
@cuda.jit("void (float64[:], float64[:], float64, int64)")
def iterate_f_kernel(d_out, d_v, a, m):
	'''
	Kernel function for evaluating m^th map iterate
	Args:
		d_out: 1D float device array of m^th iterates
		d_v: 1D float device array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		None
	'''
	i = cuda.grid(1)
	if i < d_v.shape[0]:
		d_out[i] = iterate_f(d_v[i], a, m)

def parallel_timed_array_iterate_f(v,a,m):
	'''
	Wrapper function for parallel version of timed_array_iterate_f(v,a,m)
	Compute an array of m^th iterates of the map x->f(x,a) = a(sin(PI*x))
	Args:
		v: 1D float numpy array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		result: 1D float numpy array of m^th iterates
		elapsed: float computation time in ms
	'''
	t1 = time()
	d_v = cuda.to_device(v)
	d_out = cuda.device_array_like(v)
	threads_per_block = 128
	blocks_per_grid = math.ceil(v.shape[0]/threads_per_block)
	t2 = time()
	iterate_f_kernel[blocks_per_grid, threads_per_block](d_out, d_v, a, m)
	t3 = time()
	result = d_out.copy_to_host()
	return result, 1000*(t3-t2), 1000*(t2-t1)

@cuda.jit("void (float64[:], float64[:], float64, int64)")
def gridstride_kernel(d_out, d_v, a, m):
	'''
	Gridstride kernel function for evaluating m^th map iterate
	Args:
		d_out: 1D float device array of m^th iterates
		d_v: 1D float device array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		None
	'''
	i = cuda.grid(1)
	stride = cuda.gridsize(1)
	for j in range(i, d_v.shape[0], stride):
		d_out[j] = iterate_f(d_v[j], a, m)

def gridstride_timed_array_iterate_f(v,a,m):
	'''
	Wrapper function for gridstride version of timed_array_iterate_f(v,a,m)
	Compute an array of m^th iterates of the map x->f(x,a) = a(sin(PI*x))
	Args:
		v: 1D float numpy array of initial values
		a: float parameter value
		m: int iteration count
	Returns:
		result: 1D float numpy array of m^th iterates
		elapsed: float computation time in ms
	'''
	d_v = cuda.to_device(v)
	d_out = cuda.device_array_like(v)
	threads_per_block = 128
	blocks_per_grid = math.ceil(v.shape[0]/threads_per_block)
	start = time()
	gridstride_kernel[blocks_per_grid, threads_per_block](d_out, d_v, a, m)
	elapsed = 1000*(time()-start)
	result = d_out.copy_to_host()
	return result, elapsed
	
def p2():
	'''
	Test code for problem 2
	'''
	n = 1<<22
	a = 0.2
	m = 8
	x0 = np.linspace(0,1,n)
	print("\nProblem 2")
	serial_res, serial_t = timed_array_iterate_f(x0,a,m)
	print("Serial time: ", serial_t, "ms")
	parallel_res, parallel_t_compute, t_transfer = parallel_timed_array_iterate_f(x0,a,m)
	print("Parallel compute time: ", parallel_t_compute, "ms")
	print("Parallel data transfer time: ", t_transfer, "ms")
	gridstride_res, gridstride_t = gridstride_timed_array_iterate_f(x0,a,m)
	print("Gridstride time: ", gridstride_t, "ms")
	
	print("N = ", n)
	print("Serial to parallel speedup: ", serial_t/parallel_t_compute, "X")
	print("Additional gridstride speedup: ", parallel_t_compute/gridstride_t, "X")

@cuda.jit(device=True)
def newton_update(z):
	return z - (z**3 - 1)/(3*z*z)

@cuda.jit('void (c8[: ,:] , f8[:] , f8[:], i8)')
def newton_kernel(d_out, d_x0, d_y0, m):
	'''
	Kernel function for evaluating m^th map iterate
	Args:
		d_out: 2D complex device array of m^th iterates
		d_x0: 1D float device array of initial real parts
		d_y0: 1D float device array of initial imaginary parts
		m: int iteration count
	Returns:
		None
	'''
	i,k = cuda.grid(2)
	if i < d_x0.shape[0] and k < d_y0.shape[0]:
		z = complex(d_x0[i], d_y0[k])
		for _ in range(m):
			z = newton_update(z)
		d_out[i,k] = z

def cubic_newton(x0,y0,m):
	'''
	Compute the m^th iterate of the complex map f(z)-> z**3 - 1
	Args:
		x0: 1D numpy array of float initial real parts
		y0: 1D numpy array of float imaginary real parts
		m: int iteration count
	Returns:
		2D numpy array of complex m^th iterate values
	'''
	z0 = np.zeros((x0.shape[0], y0.shape[0]), dtype=complex)
	d_x0 = cuda.to_device(x0)
	d_y0 = cuda.to_device(y0)
	d_out = cuda.to_device(z0)
	threads_per_block = (32,32)
	blocks_per_grid = (math.ceil(x0.shape[0]/threads_per_block[0]), math.ceil(y0.shape[0]/threads_per_block[1]))
	start = time()
	newton_kernel[blocks_per_grid, threads_per_block](d_out, d_x0, d_y0, m)
	elapsed = 1000*(time()-start)
	result = d_out.copy_to_host()
	return result, elapsed

def p3():
	R = 1.5 #half-width of region of interest
	NX, NY = 1000,1000 #samples along each direction
	m = 20 #iteration count
	x0 = np.linspace(-R,R,NX) #array of initial real parts
	y0 = np.linspace(-R,R,NY) #array of initial imaginary parts
	z, t = cubic_newton(x0, y0, m) #compute array of m^th newton iterates
	theta = np.angle(z)
	print("\nProblem 3")
	print("Computation time: ", t, "ms")
	plt.matshow(theta/(2*np.pi/3),
		 extent=[min(x0),max(x0),min(y0),max(y0)])
	plt.show()

def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()