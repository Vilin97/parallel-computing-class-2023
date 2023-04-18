import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

PI = np.pi
TPB = 32

def f21(x):
	'''
	Compute the value of the function in Schorghofer Problem 2.1
	Args:
		x: Float input value
	Returns float function value
	'''
	return 3 * math.pi**4 * x**2 + math.log((x-math.pi)**2)


def sample_f21(xmin, xmax, n):
    '''
    Compute a uniform sampling of the function f
    Args:
        xmin, xmax: float bounds of sampling interval
        n: int number of sample values
    Returns:
        1D numpy array of float sample values
    '''
    print("Running sample: serial version")
    x = np.linspace(xmin, xmax, n)
    y = np.zeros(n)
    for i in range(n):
        y[i] = f21(x[i])
    return y

@cuda.jit(device=True)
def f21_kernel(x):
	'''
	Compute the value of the function in Schorghofer Problem 2.1
	Args:
		x: Float input value
	Returns float function value
	'''
	return 3 * math.pi**4 * x**2 + math.log((x-math.pi)**2)

@cuda.jit
def sample_f21_parallel_kernel(d_x, d_y):
    n = d_x.shape[0]
    i = cuda.grid(1)
    if i < n:
        d_y[i] = f21_kernel(d_x[i])

def sample_f21_parallel(xmin, xmax, n):
	'''
	Parallelized computation of sample values of f_dev
	Args:
		xmin, xmax: float bounds of sampling interval
		n: int number of sample values
	Returns:
		1D numpy array of float sample values
	'''
	print("Running sample_parallel")
	x = np.linspace(xmin, xmax, n)
	y = np.zeros(n)
	d_x = cuda.to_device(x)
	d_y = cuda.to_device(y)
	gridDims = (n + TPB - 1) // TPB
	sample_f21_parallel_kernel[gridDims, TPB](d_x, d_y)
	return d_y.copy_to_host()

def p1():
	'''
	Test codes for problem 1
	'''
	xmin, xmax = 0,4
	n = 500000
	x = np.linspace(xmin, xmax, n)
	y = sample_f21(xmin, xmax, n)
	plt.plot(x, y, label='serial')

	y_par = sample_f21_parallel(xmin, xmax, n)
	diff = np.absolute(y - y_par)
	maxDiff = np.max(diff)
	print(maxDiff)
	print("Max. Diff. = ", maxDiff)
	plt.plot(x, diff, label = 'difference')
	plt.plot(x, y_par, label ='parallel')

	plt.legend()
	plt.show()

def f(x, a):
	'''
	Compute value of logistic map
	Args:
		x: float input value
		a: float parameter value
	Returns:
		float function value
	'''
	return a*x*(1-x)

def iterate(x, a, k):
    '''
    Compute an array of values for the k^th iteration of the function f(x)
    Args:
        x: numpy array of float input values
        a: float parameter value
        k: int iteration number
    Returns:
        numpy array of k^th iterate values
    '''
    n = x.shape[0]
    y = np.copy(x)
    for i in range(n):
        for j in range(k):
            y[i] = f(y[i], a)
    return y

@cuda.jit(device=True)
def f_kernel(x, a):
    return a*x*(1-x)

@cuda.jit
def iterate_parallel_kernel(d_x, d_y, a, k):
    n = d_x.shape[0]
    i = cuda.grid(1)
    if i < n:
        y = d_x[i]
        for j in range(k):
            y = f_kernel(y, a)
        d_y[i] = y

def iterate_parallel(x, a, k):
    '''
    Compute an array of values for the k^th iteration of the function f(x)
    Args:
        x: numpy array of float input values
        a: float parameter value
        k: int iteration number
    Returns:
        numpy array of k^th iterate values
    '''
    n = x.shape[0]
    y = np.copy(x)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    gridDims = (n + TPB - 1) // TPB
    iterate_parallel_kernel[gridDims, TPB](d_x, d_y, a, k)
    return d_y.copy_to_host()

def p2():
    '''
    Test codes for problem 2
    '''
    n = 10000
    k = 64
    a = 3.57
    x = np.linspace(0,1,n)
    fk = iterate(x,a,k)
    fk_par = iterate_parallel(x, a, k)
    diff = fk - fk_par
    print("Max. diff. = ", np.max(np.abs(diff)))
    plt.plot(x,x, label='identity')
    plt.plot(x,fk, label='serial')
    plt.plot(x,diff, label='difference')

    plt.legend()
    plt.show()

def cubic(x):
	return x**3 - x

def deriv(x):
	return 3*x*x - 1

def newton_update(x):
    '''
    Compute the next iterate of Newton's method
    Args:
        x: float value of current iterate
    Returns:
        Float value of next iterate
    '''
    # f(x) = f(x0) + f'(x0)(x-x0)
    # f(x) = 0
    # x = x0 - f(x0)/f'(x0)
    return x - cubic(x)/deriv(x)

def newton(x0, k_max):
    '''
    Compute the final value of a newton's method iteration after k_max steps.
    Args:
        x: float initial value
        k_max: int iteration count
    Returns:
        Float value of last iterate
    '''
    x = x0
    for i in range(k_max):
        x = newton_update(x)
    return x


def newton_serial(x0, k_max):
    '''
    Compute the last of k_max iterations of Newton's method
    Args:
        x0: 1D numpy array of float value initial values
        k_max: int iteration count
    Returns:
        1D numpy array of float values after k_max iterations
    '''
    n = x0.shape[0]
    x = np.copy(x0)
    for i in range(n):
        x[i] = newton(x0[i], k_max)
    return x

@cuda.jit(device=True)
def cubic_kernel(x):
	return x**3 - x

@cuda.jit(device=True)
def deriv_kernel(x):
	return 3*x*x - 1

@cuda.jit(device=True)
def newton_update_kernel(x):
    return x - cubic_kernel(x)/deriv_kernel(x)

@cuda.jit
def newton_parallel_kernel(d_x0, d_x, k_max):
    n = d_x0.shape[0]
    i = cuda.grid(1)
    if i < n:
        y = d_x0[i]
        for j in range(k_max):
            y = newton_update_kernel(y)
        d_x[i] = y

def newton_parallel(x0, k_max):
    '''
    Compute the last of k_max iterations of Newton's method
    Args:
        x0: 1D numpy array of float value initial values
        k_max: int iteration count
    Returns:
        1D numpy array of float values after k_max iterations
    '''
    n = x0.shape[0]
    x = np.zeros(n)
    d_x0 = cuda.to_device(x0)
    d_x = cuda.to_device(x)
    gridDims = (n + TPB - 1)//TPB
    newton_parallel_kernel[gridDims, TPB](d_x0, d_x, k_max)
    return d_x.copy_to_host()

def p3():
    '''
    Test code for problem 3.
    '''
    n = 20000
    k_max = 16
    xmin, xmax = -2,2
    x0 = np.linspace(xmin, xmax, n)
    v = newton_serial(x0, k_max)
    v_par = newton_parallel(x0, k_max)
    diff = v-v_par
    print("max diff = ", np.max(np.abs(diff)))
    plt.plot(x0, x0, label="identity")
    plt.plot(x0, v,  label=f"newton after {k_max} iterations")
    plt.legend()
    plt.show()


def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()