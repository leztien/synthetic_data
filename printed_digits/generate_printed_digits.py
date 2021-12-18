from functools import lru_cache

@lru_cache(maxsize=256)
def get_image_matrix(character, **kwargs):
    from matplotlib.pyplot import figure, text, axis, savefig, clf, close, gcf
    from matplotlib.image import imread

    fontname = ([kwargs.get(k) for k in kwargs.keys() if str(k).lower() in "fontfamily"]+["Arial"])[0]
    key = ([str(k).lower() for k in kwargs.keys() if str(k).lower() in "italicstyle"]+["italic",])[0]
    b = False if str(kwargs.get(key, 'none')).lower() in ("none","normal","false") else True
    italic = "italic" if b else "normal"
    
    figure(figsize=(1,1))
    text(0.5, 0.5, character, va='center', ha='center', size=50, fontname=fontname, style=italic)
    axis('off')

    from tempfile import TemporaryFile
    with TemporaryFile(mode='w+b') as fh:
        savefig(fh, format='jpeg', dpi=100, bbox_inches='tight', cmap='binary')
        clf()
        close(gcf())
        pn = imread(fh)
        fh.close()
    del fh

    from numpy import swapaxes
    mx = swapaxes(pn, 0,2)[0].T
    mx = pn[:,:,0]   #same

    mx = mx[::2,::2]    # skip every other row and column
    mx = (mx < 200).astype('uint8')   # binarize into 0 and 1
    return mx


def crop_image_matrix(mx, margins=3, **kwargs):
    from numpy import nonzero, mean
    nonzeros = nonzero(mx)
    xbounds = nonzeros[1].min(), nonzeros[1].max()
    ybounds = nonzeros[0].min(), nonzeros[0].max()
    xcenter = int(mean(xbounds))
    ycenter = int(mean(ybounds))

    distance_from_center = max((xbounds[1]-xbounds[0]), (ybounds[1]-ybounds[0])) // 2 + margins

    left = xcenter - distance_from_center
    right = xcenter + distance_from_center + 1
    top = ycenter - distance_from_center
    bottom = ycenter + distance_from_center + 1

    slicer = slice(top,bottom),slice(left,right)
    mx = mx[slicer]
    return mx


def rescale_image_matrix(mx, resolution=28, **kwargs):
    from numpy import zeros
    M = zeros(shape=(mx.shape[0] * resolution, mx.shape[1] * resolution), dtype='uint8')

    for i in range(mx.shape[0]):
        for j in range(0, mx.shape[1]):
            row = sum([[mx[i,j]]*resolution for j in range(mx.shape[1])],[])
            M[i*resolution : i*resolution+resolution] = row

    m = zeros(shape=(M.shape[0],resolution), dtype='float16')  # skinny, tall matrix

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i,j] = M[i, j*mx.shape[0] : j*mx.shape[0]+mx.shape[0]].mean()

    mx_rescaled = zeros(shape=(resolution,resolution), dtype='float32')
    for i in range(mx_rescaled.shape[0]):
        for j in range(mx_rescaled.shape[1]):
            mx_rescaled[i,j] = m[i*mx.shape[1] : i*mx.shape[1]+mx.shape[1], j].ravel().mean()
    return mx_rescaled


def shift_image_matrix(mx, up=None, right=None, **kwargs):
    from numpy import vstack, hstack, zeros
    vertical_steps = -(up or -kwargs.get('down', 0) or -([kwargs.get(k) for k in kwargs.keys() if str(k).lower()[:4] in ("vert","long")]+[0,])[0])
    horizontal_steps = right or -kwargs.get('left',0) or ([kwargs.get(k) for k in kwargs.keys() if str(k).lower()[:3] in ("hor","lat")]+[0,])[0]
    
    
    if vertical_steps > 0:
        mx = vstack([zeros(shape=(vertical_steps, mx.shape[1])), mx])[:mx.shape[0]]
    
    elif vertical_steps < 0:
        mx = vstack([mx, zeros(shape=(-vertical_steps, mx.shape[1]))])[-vertical_steps:]
    else: pass
    
    
    
    if horizontal_steps > 0:
        mx = hstack([zeros(shape=(mx.shape[0], horizontal_steps)), mx])[:,:mx.shape[1]]
    elif horizontal_steps < 0:
        mx = hstack([mx, zeros(shape=(mx.shape[1],-horizontal_steps))])[:,-horizontal_steps:]
    else: pass
    return(mx)


def rotate_image_matrix(mx, angle, use_scipy_function=False,  **kwargs):
    """rotate the image in the matrix around the center of the matrix"""
    from numpy import array, matrix, fliplr, zeros, clip, abs, inf
    if angle == 0: return mx
    
    if use_scipy_function:
        from scipy.ndimage.interpolation import rotate
        mx_new = rotate(mx, angle, reshape=False)
        return mx_new
    
    #the matrix image centroid
    y,x = tuple(n//2 for n in mx.shape)  
    
    from itertools import product
    nx = array(tuple(product(*[range(n)for n in mx.shape])), dtype='int16')
    values = mx.ravel()   # color values
    coordinates = nx.copy()
    
    #convert ndarray-indeces into cartesian coordinates
    coordinates[:,0] = -coordinates[:,0] + y
    coordinates[:,1] = coordinates[:,1] - x
    coordinates = fliplr(coordinates)
    M = matrix(coordinates).T   # matrix of cartesian points
    
    #transformation matrix
    from math import cos, sin, radians
    θ = radians(angle)
    T = [[cos(θ),-sin(θ)],
         [sin(θ), cos(θ)]]
    T = matrix(T, dtype='f')
    
    #matrix multiplication
    N = array((T@M).T, dtype='f') # new matrix
    
    #convert the cartesian coordinates back into ndarray-indeces
    N = fliplr(N)
    N[:,0] = -(N[:,0] - y)
    N[:,1] = N[:,1] + x
    
    #check for negative values (and shift if necessay)
    N = N + abs(clip(N.min(0), -inf, 0))
    
    #round to integers
    N = N.round().astype('int16')
    
    #make an emty new matrix
    shape = N.max(0)+1
    mx_new = zeros(shape=shape)
    
    #fill the new matrix with color values
    for value, (r,c) in zip(values, N):
        mx_new[r,c] = value
    
    #crop (because the dimensions may be larger due to rotation)
    vertical_margin = mx_new.shape[0] - mx.shape[0]
    horizontal_margin = mx_new.shape[1] - mx.shape[1]
    vertical_margin,horizontal_margin = (max(n,0)//2 for n in (vertical_margin,horizontal_margin)) #make sure there are no negative numbers
    if min(vertical_margin,horizontal_margin) > 0:
        mx_new = mx_new[vertical_margin:-vertical_margin, horizontal_margin:-horizontal_margin]
        mx_new = mx_new[-mx.shape[0]:, -mx.shape[1]:]
    
    #rescale if necessay
    if mx_new.shape != mx.shape:
        from warnings import warn
        warn("rescaling the image because the shape was {}".format(mx_new.shape), Warning)
        mx_new = rescale_image_matrix(mx_new, resolution=max(mx.shape))
    return mx_new
    
    
@lru_cache(maxsize=256)
def get_character_matrix(character, resolution=28, margins=3, flatten=False, **kwargs):
    """wrapper for get_image_matrix, crop_image_matrix, rescale_image_matrix"""
    d = locals().copy()
    d.update(d.get('kwargs',dict()))
    d.pop('kwargs') if 'kwargs' in d else None
    
    character = str(character)[0]
    d.pop('character')
    kwargs = {k:d[k] for k in d.keys() if str(k).lower() in "fontfamilyitalicstyle"}
    
    mx = get_image_matrix(character, **kwargs)
    mx = crop_image_matrix(mx, **d)
    mx = rescale_image_matrix(mx, **d)
    
    angle = ([d.get(k,0) for k in d.keys() if str(k).lower()[:3] == "rot"] + [d.get('angle',0),])[0]
    if angle:
        d['angle'] = angle
        mx = rotate_image_matrix(mx, **d)
    mx = shift_image_matrix(mx, **d)
    
    #rescale into the [0,255] range    
    mx = mx - mx.min()
    mx = (mx * (255 / mx.max())).round().astype('uint8')
    return mx.flatten() if flatten else mx


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


def make_multiple_kwargs(*args, **kwargs):
    d = args[0] if len(args)==1 and isinstance(args[0], dict) else kwargs
    assert d, "no keywords found"
    from itertools import product
    ranges = [range(len(d[k])) for k in sorted(d.keys())]
    nx = list(product(*ranges))
    return [{k:d[k][ix]  for k,ix in zip(sorted(d.keys()), t)  }  for t in nx]


def make_dataset(name, dict):
    d = dict
    kwargs = make_multiple_kwargs(d)
    m = len(kwargs)
    n = (d.get('resolution')[0] or 28)**2
    
    print("making dataset '{}' containing {} samples..".format(name, m))
    
    from numpy import empty
    X = empty(shape=(m, n), dtype='float32')
    y = empty(shape=m, dtype='uint8')   #if the target contains only integers
    y = empty(shape=m, dtype='object')  #if the target contains letters as well
    
    for i,d in enumerate(kwargs):
        X[i] = get_character_matrix(**d)
        y[i] = d['character'] 
        
        if i>0 and i%100==0:
            print("..sample {} generated".format(i))
        
    make_dataset.name = name
    X.flags.writeable = False
    y.flags.writeable = False
    return X,y

#==================================================================================================


"""MAKE DATASETS"""
def make_3_datasets():
    #define kwargs for the three datasets
    from string import ascii_letters
    fonts = ["Garamond","Lucida Console","Sylfaen","Tahoma","Corbel","Cambria","DejaVu Sans", "Arial", "Times New Roman","Consolas","Calibri","Candara","Century"]
    
    standard = dict(character=[*range(10)]*2,                                # +[*ascii_letters]
             font=fonts,
             italic=[False,],
             rot=[0,],
             down=[0,],
             right=[0,],
             margins=[2,],
             resolution=[28,],
             flatten=[True,])
    
    augmented = dict(character=[*range(10)],                                # +[*ascii_letters]
             font=fonts,
             italic=[False,],
             rot=[0,],
             down=[0,-1, 1],
             right=[0,-1, 1,],
             margins=[2,],
             resolution=[28,],
             flatten=[True,])
    
    rotated = dict(character=[*range(10)],                                # +[*ascii_letters]
             font=fonts,
             italic=[False,True],
             rot=[0, 10, -5],
             down=[0,],
             right=[0,],
             margins=[2,],
             resolution=[28,],
             flatten=[True,])
    
    #MAKE THE BUNCH-DICTIONARY (with the 3x2=6 ndarrays)
    names = "standard augmented rotated".split()

    bunch = dict()
    for name in names:
        kwargs = eval(name)
        X,y = make_dataset(name, kwargs)
        temp = dict(data=X, target=y)
        bunch[name] = temp
    #add description to the bunch-dictionary
    s = ("This is a 3-piece collection of datasets of printed digits\n"
         "standard dataset: non-modified printed digits\n"
         "augmented dataset: digits shifted one pixel up/down/left/right\n"
         "rotated dataset: digits rotated ca. 10 degrees clockwise and counter-clockwise\n"
         "Notes:\n"
         "the datasets are not shuffled and not split into training/test sets\n"
         "the standard dataset contains a duplicate of itself to compensate for the low number of samples\n"
         "the augmented dataset contains the standard dataset in itself\n"
         "the rotated dataset also contains the standard dataset in itself, as well as italicized versions of digits")
    bunch["description"] = s
    
        
    #PICKLE THE BUNCH-DICTIONARY
    FILENAME = r"printed_digits.pkl"
    import pickle
    with open(FILENAME, mode='wb') as fh:
        pickle.dump(bunch, file=fh)
    print("done")



def make_rotated_dataset(name="rotated"):
    from string import ascii_letters, ascii_uppercase
    fonts = ["Garamond","Lucida Console","Sylfaen","Tahoma","Corbel","Cambria","DejaVu Sans", "Arial", "Times New Roman","Consolas","Calibri","Candara","Century"]
    
    params = dict(character=list(range(10)) + list(ascii_uppercase),                                # +[*ascii_letters]
             font=fonts,
             italic=[False,True],
             rot=[0, 10, -5, 25, -20],
             down=[0,],
             right=[0,],
             margins=[3,],
             resolution=[28,],
             flatten=[True,],
             use_scipy_function=[True,])
    
    #params = dict(character=["A","b", 3], font = ["Garamond","Lucida Console"],resolution=[28,])
    
    #MAKE THE BUNCH-DICTIONARY
    X,y = make_dataset(name, params)
    bunch = dict(data=X, target=y)   
        
    #PICKLE THE BUNCH-DICTIONARY
    FILENAME = r"{}.pkl".format(name)
    import pickle
    with open(FILENAME, mode='wb') as fh:
        pickle.dump(bunch, file=fh)
    print("done creating the '{}' dataset".format(name))

#===========================================================================================================



def main():
    make_3_datasets()
    
    import pickle
    FILENAME = r"printed_digits.pkl"
    with open(FILENAME, mode='rb') as fh:
        bunch = pickle.load(fh)
    globals()['bunch'] = bunch
#if __name__=="__main__":main()


