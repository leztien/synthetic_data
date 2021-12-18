


def straighten_up_character(mx:'2D ndarray representing a character image', 
                            return_angle:'return the angle instead of transformed matrix'=False):
    """    Put an inclined symbol in a 2D-ndarray-image straight. 
    Uses eigenvectors to do that.
    Get either the matrix with the straightened up symbol or the angle to use with scipy.ndimage.interpolation.rotate"""
    
    from itertools import product
    from numpy import ndarray, array, clip, logical_or, c_, cov, argmax, dot, zeros
    from numpy.linalg import eig
    if not (isinstance(mx, ndarray) and mx.ndim==2): raise TypeError("must be a 2D ndarray")

    #make an index array representing the matrix's indeses
    nx = list(product(range(mx.shape[0]), range(mx.shape[1])))   # ndarray-indeces  (row,column) format
    nx = array(nx, dtype='i')
    
    #colour values in the unravelled form (each pocket corresponds to the index in the nx array)
    values = mx.flatten()
    
    #omit zero values i.e. white pixels
    mask = values > 0
    nx,values = (a[mask] for a in (nx,values))
    
    #FIND THE CENTER OF THE CHARACTER (not the center of the matrix) 
    B = clip(mx, 0,1).astype(bool)   # boolean matrix denoting coloured pixels
    X = logical_or.reduce(B, axis=0) # boolean x-axis
    Y = logical_or.reduce(B, axis=1) # boolean y axis
    
    #find where the character colour pixels begin and end
    try: xstart = X.tolist().index(True)
    except ValueError: xstart = 0
        
    try: ystart = Y.tolist().index(True)
    except ValueError: ystart = 0
    
    try: xend = (mx.shape[1]-1) - list(reversed(X)).index(True)
    except ValueError: xend = mx.shape[1]-1
    
    try: yend = (mx.shape[0]-1) - list(reversed(Y)).index(True)
    except ValueError: yend = mx.shape[0]-1
    
    x_center = (xend - xstart)//2 + xstart
    y_center = (yend - ystart)//2 + ystart
    
    #convert array-indeces into cartesian coordinates
    xcoords = nx[:,1] - x_center
    ycoords = -(nx[:,0] - y_center)
    coords = c_[xcoords,ycoords]
    
    #get covariance matrix
    Σ = cov(coords.T)   #covariance matrix
    
    #get the eigenvectors
    λλ,ee = eig(Σ)   # eigenvalues and unit i.e. normalized eigenvectors
    e1,e2 = ee[:,argmax(λλ)], ee[:,int(not bool(argmax(λλ)))]
    e1 = -e1    # change the direction of the vector (arbitrary)
    e2 = -e2
    
    #get the angle of the eigenvector
    x = e1[0]
    from math import asin, degrees
    angle = degrees(asin(x))
    straighten_up_character.angle = angle
    if return_angle: return angle
    
    #calculate the projections on the eigenvectors (via dotproduct)
    ycoords = dot(coords, e1)
    xcoords = dot(coords, e2)
    
    #convert Cartesian coordinates back into array-indeces
    xcoords = xcoords + x_center
    ycoords = -ycoords + y_center
    coords = c_[ycoords,xcoords]
    coords = coords.round()
    
    #provide for the situation that the new matrix streaches beyond the original one and therefore might have now negative values for array-indeses
    mask = ~logical_or(coords[:,0]<0, coords[:,1]<0)  #what if some indeces ended up with negative values?...
    coords = coords[mask].astype('uint16')   #eliminate those pixels and indeses
    values = values[mask]
    
    #make a new matrix and fill the new empty matrix with values
    mx_new = zeros(shape=c_[coords.max(0), mx.shape].max(1), dtype='uint8')
    nx_x, nx_y = coords.T
    mx_new[nx_x, nx_y] = values
    return mx_new



def main():

    from generate_printed_digits import get_character_matrix
    from matplotlib.pyplot import matshow, show
    from scipy.ndimage.interpolation import rotate
    
    #DATA
    mx = get_character_matrix("5", resolution=50)
    mx = rotate(mx, angle=-25, reshape=False)
    
    mx_new = straighten_up_character(mx)
    angle = straighten_up_character(mx, return_angle=True)
    
    matshow(mx_new, cmap='binary')
    
    mx2 = rotate(mx, angle, reshape=False)
    
    matshow(mx2, cmap='binary')
    
    #show()

if __name__=="__main__":main()















