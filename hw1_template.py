import numpy as np
import matplotlib.pyplot as plt

def f(x, a):
    '''
    Logistic map
    Args:
        a: float parameter value
        x: float input value
    Returns:
        float value of logistic map function
    '''
    return a*x*(1-x)

def trajectory(f, x0, a, n): 
    '''
    Compute an array of successive map iterates
    Args:
        f: name of mapping function
        x0: float initial value
        a: float parameter value
        n: int number of iterations
    Returns:
        float numpy array of successive iterate values
    '''
    trajectory = np.zeros(n)
    trajectory[0] = x0
    for i in range(1,n):
        trajectory[i] = f(trajectory[i-1], a)
    return trajectory

def spiderize(v):
    '''
    Compute arrayof points for spiderweb plot
    Args:
        v: float numpy array of successive iterates
    Returns:
        x_spider: float numpy array of x coordinates
        y_spider: float numpy array of y coordinates
    '''
    x_spider = np.zeros(2*v.shape[0]-2)
    y_spider = np.zeros(2*v.shape[0]-2)
    for i in range(v.shape[0]-1):
        x_spider[2*i] = v[i]
        x_spider[2*i+1] = v[i+1]
        y_spider[2*i] = v[i+1]
        y_spider[2*i+1] = v[i+1]
    return x_spider, y_spider

def iterate(f, x, a, k):
    '''
    Compute an array of values for the k^th iteration of a mapping function
    Args:
        f: name of mapping function
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
    Test code for problem 2
    '''
    print("Start p2().")
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values
    m = 21 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]
    for j in range(2): # for the first 2 parameter values
        a = param[j]
        y = np.zeros(m) #array for storing computed values of the map
        for i in range(m): #for each entry in the array x
            y[i] =f(x[i],a) #store map value in corresponding entry in array y
        #visualize the map
        plt.plot(x,x) #45-degree line
        plt.plot(x,y) #map data
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('F')
        plt.show() #show the plot
    print("How did the plot change?") 
    print('How will that affect "steady-state" behavior?')
    print('Inspect each of the following plots and notice how steady-state behavior changes.')

    n = 1<<6 #trajectory length
    for j in range(param.shape[0]):
        a = param[j]
        x0 = 0.2
        iter_history = trajectory(f, x0, a, n)
        plt.plot(iter_history)
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('Iterations')
        plt.ylabel('$X_k$')
        plt.show() #show the plot

def p3():
    print("Start p3().")
    m = 21 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values

    n = 1<<6 #trajectory length
    for j in range(param.shape[0]):
        a = param[j]
        x0 = 0.2
        iter_history = trajectory(f, x0, a, n)
        spider_x, spider_y = spiderize(iter_history)
        plt.plot(x,x) #45-degree line
        plt.plot(spider_x[6:-1], spider_y[6:-1]) #plot behavior after transient
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('$F(X)$')
        plt.show() #show the plot

def p4():
    print("Start p4().")
    k = 16 #sample iteration number
    m = 1024 # number of sample points
    x = np.linspace(0,1,m) #array of equally space points on [0,1]
    param = np.array([0.9, 1.9, 2.9, 3.1, 3.5, 3.57]) #example parameter values

    for j in range(param.shape[0]):
        a = param[j]
        kth_iter = np.zeros_like(x)
        kth_iter = iterate(f, x, a, k )
        plt.plot(x,x) #45-degree line
        plt.plot(x, kth_iter) #plot behavior after transient
        plt.ylim([0,1])
        #label your axes (so your plot is not "information free")
        plt.xlabel('X')
        plt.ylabel('$f^k(X)$')
        plt.show() #show the plot


if __name__ == '__main__':
    # p2()
    p3()
    # p4()