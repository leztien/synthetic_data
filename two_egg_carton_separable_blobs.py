
"""
function two_egg_carton_separable_blobs
two n-dimensional blobs meshed together and separable by an imaginary egg-carton
the separation boundary is formed by sin/cos (hyper)curve
"""

def rotation_matrix(n=3):
    """
    returns a rotation matrix for rotation in n dimensions
    the angle of rotation in each hyperplane is chosen randomly
    bug: sign in from of the sin() is not correct (must be swapped, but IDK the principle) however the lengths of all baseis vectors is 1
    """
    import numpy as np
    from itertools import combinations
    from functools import reduce

    t = tuple(combinations(range(n), r=2))
    n_rotations = len(t)

    RR = np.zeros(shape=(n_rotations,n,n))
    [np.fill_diagonal(RR[h], 1) for h in range(n_rotations)]

    angles = np.deg2rad(np.random.randint(-90,90, size=len(RR)))
    angles = (np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]]) for angle in angles)

    for R,t,angle in zip(RR,t,angles):
        nx = np.ix_(t,t)
        R[nx] = angle

    R = np.array(reduce(np.matmul, RR))
    return(R)


def separate_by_curve(X):
    """separates an n-dimensional data-blob with a sin/cos (hyper)curve along the last axis"""
    from numpy import sin,cos,ceil,greater
    n = X.shape[1]
    frequency = 4   # keep it constant
    amplitude = 0.5
    n = X.shape[-1]
    XX = X[:,:-1] * frequency
    functions = sum([[sin, cos] for _ in range(int(ceil(n/2)))],[])[:n-1]

    for j in range(n-1):
        XX[:,j] = functions[j](XX[:,j]) * amplitude
    y = greater(X[:,-1], XX.sum(1)).astype("uint8")
    return(y)


def two_egg_carton_separable_blobs(m=100, n=3):
    """
    two n-dimensional blobs meshed together and speraable by an imaginary egg-carton
    the separation boundry is formed by sin/cos (hyper)curve
    """
    from numpy import eye, matmul, log
    from numpy.random import randn
    X = randn(m,n)

    #squishing transformation matrix (squish the blob along the last axis)
    S = eye(n)
    S[-1,-1] = 1.5
    X = matmul(S, X.T).T
    y = separate_by_curve(X)

    #separate the two sections by shifting the upper one upwards
    gap = (X[:,-1].max() - X[:,-1].min()) / 10
    gap = gap * log(n)
    X[y==1, -1] += gap

    #rotate the blob
    R = rotation_matrix(n=n)
    X = matmul(R, X.T).T

    #shift into the positive qadrant
    X -= X.min(0)
    return(X,y)


##########################################################################


def demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from pandas.plotting import scatter_matrix
    from seaborn import pairplot

    X,y = two_egg_carton_separable_blobs(m=1000, n=5)

    colnames = ['x{}'.format(i) for i in range(X.shape[1])]+['y']
    df = DataFrame(np.hstack([X,y[:,None]]), columns=colnames)

    try: pairplot(df, hue='y', markers='.')
    except: scatter_matrix(df.iloc[:,:-1], )
    plt.show()
if __name__=='__main__':demo()
