

"""
make n-dimensional dataset with k lineraly seperable classes (for classification problems)
"""

import numpy as np


def make_blob(m, n, random_state=None, return_radius=False):
    rs = random_state or globals().get("rs", np.random.randint(1,999))
    rs = rs if isinstance(rs, np.random.mtrand.RandomState) else np.random.RandomState(int(rs))

    sigmas = rs.chisquare(df=3, size=n)**0.5 if bool(rs.randint(0,2)) else rs.uniform(low=0.1, high=3, size=n)
    blob = rs.normal(loc=0, scale=sigmas, size=(m,n))
    blob = blob - blob.mean(0)  # center the blob
    ix = (blob**2).sum(1).argmax()
    radius = np.sqrt((blob[ix]**2).sum())
    return (blob,radius) if return_radius else blob


def make_rotation_matrix(n, n_rotations=None, max_angle=None, random_state=None):
    """rotation matrix in n-dimensions"""
    rs = random_state or globals().get("rs", np.random.randint(1,999))
    rs = rs if isinstance(rs, np.random.mtrand.RandomState) else np.random.RandomState(int(rs))

    if n_rotations==0: return np.eye(n)
    from itertools import combinations
    plane_combinations = tuple(combinations(range(n), r=2))  # 2d-plane combinations

    if n_rotations==0: return np.eye(n)
    if n_rotations != "all":  # randomly select wich planes to rotate on
        n_rotations = n_rotations or rs.randint(1, len(plane_combinations))
        n_rotations = int(n_rotations)
        assert 0 < n_rotations <= n*(n-1)/2,"err"
        nx = np.sort(rs.permutation(len(plane_combinations))[:n_rotations])
        assert len(nx)>=1 and max(nx)<len(plane_combinations),"err"
        plane_combinations = np.array(plane_combinations)[nx]

    #create lists of id-matreces, angles
    max_angle = max_angle or 180
    III = [np.eye(n) for _ in plane_combinations]
    θθθ = np.deg2rad(rs.randint(-max_angle, max_angle, size=len(plane_combinations)))
    trigs = [np.array([np.cos(θ), -np.sin(θ), np.sin(θ), np.cos(θ)]).reshape(2,2)
                for θ in θθθ]

    for ix,I,trig in zip(plane_combinations, III, trigs):
        nx = np.ix_(ix,ix)
        I[nx] = trig

    from functools import reduce
    T = reduce(np.matmul, III)
    return T


def make_data_for_classification(m:'total number of data-points',
                                 n:'number of dimensions/features',
                                 k:'number of classes',
                                 blobs_density:'ratio denoting the relative vicinity of the blobs to the central blob' = None,
                                 random_state=None):
    """make n-dimensional linearly seperable data with k classes.
    The structure of the data and how it fills the space can be visualized in 3d (if n==3)
    (the data is unnormalized/unstandardized and is located in the positive hyper-quadrant)"""

    n_points_total = m  # n_points_total will keep the total number of data-point for later
    m = m // k          # number of data-points per class
    blobs_density = blobs_density or 0.5   # density of blobs (0, 1)

    #random state
    rs = random_state or globals().get("rs", np.random.randint(1,999))
    rs = rs if isinstance(rs, np.random.mtrand.RandomState) else np.random.RandomState(int(rs))


    # make k blobs with respective radii
    blobs_w_radii = [make_blob(m, n, random_state=rs, return_radius=True) for _ in range(k)]
    blobs = [t[0] for t in blobs_w_radii]
    radii = [t[1] for t in blobs_w_radii]

    #make k rortation matreces and rotate each blob with the respective rotation matrix
    rotations = [make_rotation_matrix(n, max_angle=45, random_state=rs) for _ in range(k)]
    blobs = [(R@M.T).T for R,M in zip(rotations,blobs)]

    #make k-1 unit-vectors pointing in random directions in the n-dimensional space
    transformations_for_unit_vector = [make_rotation_matrix(n, n_rotations='all', random_state=rs) for _ in range(k)]
    v = np.array([1, *[0,]*(n-1)]).reshape(-1,1)  # i-hat basis-vector
    vectors = [T@v for T in transformations_for_unit_vector]  # vector[0] will be ignored

    #shift the k-1 blobs in the direction of the respective random unit vector
    for i in range(1,k):
        vector = vectors[i] * (radii[0] + radii[i] * blobs_density)
        blobs[i] = blobs[i] + vector.flatten()

    #make the target vector, concatinate the data
    y = sum(([label,]*len(blob) for blob,label in zip(blobs, range(k))), [])
    mx = np.concatenate(blobs, axis=0)
    mx = mx + mx.min(axis=0).__abs__()

    #add missing points to make the total number of data-point equal the number provided by the user
    n_missing_points =  n_points_total - len(mx)
    X = np.concatenate([mx, mx[-n_missing_points:]+0.001], axis=0)
    y.extend(y[-n_missing_points:])

    y = np.array(y, dtype='uint8')
    assert len(X)==len(y),"err3"
    return(X,y)

#====================================================================================


def visualize(visualize_tSNE=True):
    X,y = make_data_for_classification(m=500, n=3, k=5, blobs_density=0.7)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12,5))
    grid = 121 if visualize_tSNE else 111
    sp = fig.add_subplot(grid, projection="3d")
    sp.scatter(*X.T, c=y)
    
    if visualize_tSNE:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        Xtsne = tsne.fit_transform(X)
        sp = fig.add_subplot(122)
        sp.scatter(*Xtsne.T, c=y, marker='.')
    
    plt.show()
    
    
    #build classification model
    from sklearn.linear_model import LogisticRegression
    
    X,y = make_data_for_classification(m=1000, n=100, k=12, blobs_density=0.2)
    md = LogisticRegression(fit_intercept=True, solver='sag', multi_class='multinomial', C=100, penalty='l2', max_iter=9999)
    md.fit(X,y)
    accuracy = md.score(X,y)
    print("accuracy =", accuracy)
    

if __name__=='__main__':visualize(visualize_tSNE=True)

