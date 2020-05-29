import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_swiss_roll():
    """swiss roll (with noise + rotated)"""
    m = 100
    t = np.linspace(0, 15*np.pi, m) ** (2/3)
    x = np.cos(t) * t #**.5
    y = np.sin(t) * t #**.5
    
    #calculate the z-step
    x,y,t = (nd[m//6:] for nd in (x,y,t))  #drop off the first points (for aesthetic reasons)
    step = ((x[m//2] - x[m//2-1])**2 + (y[m//2] - y[m//2-1])**2)**.5 * 0.8
    z = np.arange(start=0, stop=10*step*2, step=step)
    
    #make a panel
    pn = np.empty(shape=(len(z), len(t), 4))
    pn[:, :, 0] = t
    pn[:, :, -1] = z.reshape(-1,1)
    X = np.vstack(pn)  # unstack the panel into a 2d matrix
    
    #add noise to t
    X[:,0] += np.random.normal(loc=0, scale=0.05, size=X.shape[0])
    t = X[:,0]
    X[:,1] = np.cos(t) * t
    X[:,2] = np.sin(t) * t
    
    #add noise to axis z
    X[:,-1] += np.random.normal(loc=0, scale=0.2, size=X.shape[0])
    
    #drop the t column
    X = X[:, 1:]
    
    #make y
    y = X[:,0]**2 + X[:,1]**2
    y = y / y.max()
    
    #rotate the roll
    from math import sin, cos, radians
    θ = radians(30)
    M = [(cos(θ),-sin(θ)),(sin(θ), cos(θ))]
    Rx,Ry,Rz = (np.eye(3) for _ in range(3))
    Rx[1:, 1:] = M
    Ry[::2, ::2] = np.array(M).T
    Rz[:2, :2] = M
    R = Rz @ Ry @ Rx
    X = np.matmul(R, X.T).T
    return(X,y)


######################################################################



X,y = make_swiss_roll()


sp = plt.axes(projection="3d")
sp.scatter(*X.T, '.', c=y, cmap="hot")
plt.show()
