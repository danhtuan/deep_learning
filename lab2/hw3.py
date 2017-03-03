import matplotlib
matplotlib.use('gtk')
import matplotlib.pyplot as plt
import numpy as np

R = 1
Q = 3
lr = 0.01

x = np.zeros((2,1))
P = np.array([-1, 0, 1]).reshape(3,1)
T = np.array([-1.5, 0, 2.5]).reshape(3,1)
G = np.array([[-1,0,1], [1,1,1]]).T

c = np.dot(T.T, T)
d = np.dot(G.T, T) * (-2)
A = np.dot(G.T, G) * 2

def floss():
   F = np.dot(np.dot(x.T, A), x) 
   F = F + np.dot(d.T, x)
   F = F + c
   return F

def df():
    delta = np.dot(A, x)
    delta = delta + d
    return delta

def update(grad):
    global x
    x = x - grad * lr

sse = np.zeros(1000) 

def runtest():
    global sse
    iter = 0
    sse[iter] = floss()
    grad = df()
    print('iter[{0}]: x={1} floss={2} grad={3}'.format(iter, x, floss(), np.linalg.norm(grad)))
    while (np.linalg.norm(grad) >= 0.01) & (iter < 999):
        iter = iter + 1
        grad = df()
        update(grad)
        sse[iter] = floss()
        print('iter[{0}]: x={1} floss={2} grad={3}'.format(iter, x, floss(), np.linalg.norm(grad)))
    plt.plot(sse[0:iter])   
runtest()
    
        
	
