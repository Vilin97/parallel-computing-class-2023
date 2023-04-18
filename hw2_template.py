import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

N = 64000
PI = np.pi
TPB = 32

def f21(x):
	'''
	Compute the value of the function in Schorghofer Problem 2.1
	Args:
		x: Float input value
	Returns float function value
	'''
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


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
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)

'''
INSERT YOUR CODE HERE TO SUPPORT sample_f21_parallel
'''


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
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


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
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


'''
INSERT YOUR CODE HERE TO SUPPORT iterate_parallel
'''

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
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


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
	plt.plot(x,x)
	plt.plot(x,fk)
	plt.plot(x,diff)
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
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


def newton(x0, k_max):
	'''
	Compute the final value of a newton's method iteration after k_max steps.
	Args:
		x: float initial value
		k_max: int iteration count
	Returns:
		Float value of last iterate
	'''
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


def newton_serial(x0, k_max):
	'''
	Compute the last of k_max iterations of Newton's method
	Args:
		x0: 1D numpy array of float value initial values
		k_max: int iteration count
	Returns:
		1D numpy array of float values after k_max iterations
	'''
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)


'''
INSERT YOUR CODE HERE TO SUPPORT newton_parallel
'''

def newton_parallel(x0, k_max):
    '''
	Compute the last of k_max iterations of Newton's method
	Args:
		x0: 1D numpy array of float value initial values
		k_max: int iteration count
	Returns:
		1D numpy array of float values after k_max iterations
	'''
	#INSERT YOUR CODE HERE
    pass #REMOVE THIS LINE (which is just here to prevent an error message)

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
	plt.plot(x0,x0)
	plt.plot(x0, v)
	plt.show()


def main():
	p1()
	p2()
	p3()

if __name__ == '__main__':
	main()