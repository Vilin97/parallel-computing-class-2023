import numpy as np
import matplotlib.pyplot as plt
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
    #INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass

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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass
	
    
@cuda.jit
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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass

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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass

@cuda.jit
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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass

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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass
	
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

@cuda.jit
def newton_kernel(d_out, d_x0, d_y0, m):
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass
	
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
	#INSERT YOUR CODE HERE AND DELETE THE LINE BELOW
	pass

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