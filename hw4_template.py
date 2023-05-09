import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit
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
	#INSERT CODE HERE
	pass

@cuda.jit
def dist_kernel(d_out, d_x, d_y, t_f, h):
	pass

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
	#INSERT CODE HERE
	pass

def p1():
	print("\nProblem 1")
	RX,RY = 1.,1.
	NX,NY = 1024, 1024
	t_f = 100
	h = 0.01
	x = np.linspace(-RX,RX,NX, dtype = np.float32)
	y = np.linspace(-RY,RY,NY, dtype = np.float32)
	d = dist(x,y,t_f,h)
	plt.imshow(d, extent=[-RX,RX,-RY,RY], vmin=0., vmax = 2.5)
	cb = plt.colorbar()
	plt.title('Distance from Equilibrium')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.show()

@cuda.jit		
def parallel_kernel(d_out, d_data, r):
	#INSERT CODE HERE
	pass

def parallel_weekly(data, r):
	'''
	Compute a moving average with radius r (global memory version)
		data: 1D numpy float array
		r: int radius
	Returns:
		1D numpy array of average over 2*r+1 consectutive values.
	'''
	#INSERT CODE HERE
	pass

@cuda.jit		
def shared_weekly_kernel(d_out, d_data, r):
	pass

def shared_weekly(data, r):
	'''
	Compute a moving average with radius r (shared memory vesion)
		data: 1D numpy float array
		r: int radius
	Returns:
		1D numpy array of average over 2*r+1 consectutive values.
	'''
	#INSERT CODE HERE
	pass

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
	
	r = 2

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
	#INSERT CODE HERE
	pass

@cuda.jit
def simpson_kernel(d_contribs, d_v):
	#INSERT CODE HERE
	pass

def par_simpson(v, h):
	'''
	Compute composite Simpson's quadrature estimate from uniform function sampling
	Args:
		v: 1D float numpy array of sampled function values
		h: float sample spacing
	Return:
		Float quadrature estimate
	'''
	#INSERT CODE HERE
	pass

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
