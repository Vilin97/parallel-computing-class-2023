import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit, float32
import math
from time import time
import csv

EPS = 0.1
RAD = 1
TX, TY = 16, 16
TPB = 512

@cuda.jit(device=True)
def accel(x,y):
	'''
	Compute the rate of coordinate change for the vdP eqn
	Args:
		x,y: float phase-plane coords. (displacement and velocity)
	Returns: 
		float acceleration
	'''
	return -x + EPS*(1-x*x)*y

@cuda.jit(device=True)
def euler_dist(x0, y0, t_f, h):
	'''
	Use Euler's method to compute the solution of the van der Pol equation
	Args:
		x0, y0: float initial displacement and velocity
		t_f: float time interval
		h: float time step
	Returns:
		Float final distance from origin of phase plane: sqrt(x*x+y*y)
	'''
	x = x0
	y = y0
	t = 0
	while t < t_f:
		x = x + h*y
		y = y + h*accel(x,y)
		t = t + h
	return math.sqrt(x*x+y*y)

@cuda.jit
def dist_kernel(d_out, d_x, d_y, t_f, h):
	i,j = cuda.grid(2)
	if i < d_out.shape[0] and j < d_out.shape[1]:
		d_out[i,j] = euler_dist(d_x[i], d_y[j], t_f, h)

def dist(x, y, t_f, h):
	'''
	Compute solutions of vdP equation for a grid of initial conditions
	Args:
		x,y: 1D arrays of initial displacement and velocity
		t_f: float time interval
		h: float time step
	Returns:
		2D numpy float array of final distance from origin of phase plane
	'''
	out = np.zeros((x.shape[0], y.shape[0]), dtype = np.float32)
	d_x = cuda.to_device(x)
	d_y = cuda.to_device(y)
	d_out = cuda.to_device(out)
	threads_per_block = (32,32)
	blocks_per_grid = (math.ceil(x.shape[0]/threads_per_block[0]), math.ceil(y.shape[0]/threads_per_block[1]))
	dist_kernel[blocks_per_grid, threads_per_block](d_out, d_x, d_y, t_f, h)
	return d_out.copy_to_host()

def p1():
	print("\nProblem 1")
	RX,RY = 1.,1.
	NX,NY = 1024, 1024
	t_f = 100
	h = np.float32(0.01)
	x = np.linspace(-RX,RX,NX, dtype = np.float32)
	y = np.linspace(-RY,RY,NY, dtype = np.float32)
	d = dist(x,y,t_f,h)
	plt.imshow(d, extent=[-RX,RX,-RY,RY], vmin=0., vmax = 2.5)
	cb = plt.colorbar()
	plt.title('Distance from Equilibrium')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.show()
	plt.savefig('p1.png')

@cuda.jit		
def parallel_kernel(d_out, d_data, r):
	i = cuda.grid(1)
	if i < d_out.shape[0]:
		avg = 0
		for j in range(-r,r+1):
			avg += d_data[min(max(i+j,0),d_data.shape[0]-1)]
		d_out[i] = avg/(2*r+1)

def parallel_weekly(data, r):
	'''
	Compute a moving average with radius r (global memory version)
		data: 1D numpy float array
		r: int radius
	Returns:
		1D numpy array of average over 2*r+1 consectutive values.
	'''
	d_data = cuda.to_device(data)
	d_out = cuda.device_array_like(data)
	threads_per_block = TPB
	blocks_per_grid = math.ceil(data.shape[0]/threads_per_block)

	e_start = cuda.event()
	e_end = cuda.event()
	e_start.record()
	parallel_kernel[blocks_per_grid, threads_per_block](d_out, d_data, r)
	e_end.record()
	e_end.synchronize()
	event_time = cuda.event_elapsed_time(e_start, e_end)
	return d_out.copy_to_host(), event_time

@cuda.jit		
def shared_weekly_kernel(d_out, d_data, r):
	n = d_data.shape[0]
	i = cuda.grid(1)
	shared_array = cuda.shared.array(shape=516, dtype=float32)
	tx = cuda.threadIdx.x
	sx = tx + r
	if i>=n: 
		return #bounds check
	shared_array[sx] = d_data[i] # regular cells
	if tx < r: # the first r threads load the halo cells
		if i>= r:
			shared_array[sx - r] = d_data[i-r]
		if i + cuda.blockDim.x < n:
			shared_array[sx+cuda.blockDim.x] = d_data[i+cuda.blockDim.x]
	cuda.syncthreads()
	# if tx == 0 and i < 4:
	# 	print("Block index: ", cuda.blockIdx.x)
	# 	print("Thread index: ", tx)
	# 	print("Shared index: ", sx)
	# 	print("Shared array[1]: ", shared_array[1])
	# 	print("Data index: ", i)
		
	if i < d_out.shape[0]:
		avg = 0
		for j in range(-r,r+1):
			avg += shared_array[sx+j]
		d_out[i] = avg/(2*r+1)

def shared_weekly(data, r):
	'''
	Compute a moving average with radius r (shared memory vesion)
		data: 1D numpy float array
		r: int radius
	Returns:
		1D numpy array of average over 2*r+1 consectutive values.
	'''
	d_data = cuda.to_device(data)
	d_out = cuda.device_array_like(data)
	threads_per_block = TPB
	blocks_per_grid = math.ceil(data.shape[0]/threads_per_block)

	e_start = cuda.event()
	e_end = cuda.event()
	e_start.record()
	shared_weekly_kernel[blocks_per_grid, threads_per_block](d_out, d_data, r)
	e_end.record()
	e_end.synchronize()
	event_time = cuda.event_elapsed_time(e_start, e_end)
	return d_out.copy_to_host(), event_time

def p2():
	def str_to_float(string):
		return float(string.replace(",",""))

	filename = "dj_data.txt"

	col = -1
	delimiter = "	"

	print("\n\nProblem 2")
	#make sure dj_data.txt is saved to same folder as this file
	# read data values from file
	vals = []
	with open(filename) as csvfile:
		file = csv.reader(csvfile, delimiter = delimiter)
		for row in file:
			vals.append(str_to_float(row[col]))
	
	r = 0

	N = 10000 #repeat the data to create large data set
	vals = np.array(N*vals, dtype = np.float32)

	par_avg, par_t = parallel_weekly(vals,r) #untimed
	par_avg, par_t = parallel_weekly(vals,r)

	sh_avg, sh_t = shared_weekly(vals,r) #untimed
	sh_avg, sh_t = shared_weekly(vals,r)
	#check that results match
	# print(vals[:10])
	# print(par_avg[:10])
	# print(sh_avg[:10])
	print("max diff: ", np.max(np.abs(sh_avg - par_avg)))
	print("Speedup due to shared: ", par_t/sh_t,"X")

@cuda.reduce
def sum_reduce(a, b):
	return a+b

@cuda.jit
def simpson_kernel(d_contribs, d_v):
	i = cuda.grid(1)
	if i < d_contribs.shape[0]:
		d_contribs[i] = d_v[2*i] + 4*d_v[2*i+1] + d_v[2*i+2]

def par_simpson(v, h):
	'''
	Compute composite Simpson's quadrature estimate from uniform function sampling
	Args:
		v: 1D float numpy array of sampled function values
		h: float sample spacing
	Return:
		Float quadrature estimate
	'''
	d_v = cuda.to_device(v)
	contribs = np.zeros(v.shape[0]//2, dtype = np.float32)
	d_contribs = cuda.to_device(contribs)
	threads_per_block = TPB
	blocks_per_grid = math.ceil(d_contribs.shape[0]/threads_per_block)
	simpson_kernel[blocks_per_grid, threads_per_block](d_contribs, d_v)
	return sum_reduce(d_contribs) * h/3

def p3():
	print("\n\nProblem 3")
	n = 100000
	xmin, xmax = 0, 2*np.pi
	h = (xmax-xmin)/(n-1)
	x = np.linspace(xmin, xmax, n)
	v = np.sin(100*x) * np.sin(100*x)
	result = par_simpson(v, h)
	print("Integral value: ", result)

def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()
