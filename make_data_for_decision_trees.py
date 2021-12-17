
"""
make data for decision trees  (the code needs tweaking)
"""


import numpy as np
from random import random, uniform, shuffle, randint


def make_data_for_decision_trees(m, n, k, proportion_of_binary_features = None):
    
    # Arbitrary values fro max_depth and min_points_in_a_leaf
    max_depth = np.log(m) * n * k / 10
    min_leaf = np.log(m) * n * k / 25
    
    if not (isinstance(proportion_of_binary_features, (float,int)) and (0 <= proportion_of_binary_features <= 1)):
        proportion_of_binary_features = uniform(0.2, 0.8)  # instead of beta distribution
    
    n_binary_features = round(n * proportion_of_binary_features)
    n_continuous_features = n - n_binary_features
    
    binary_features = list()
    for j in range(n_binary_features):
        p = uniform(0.2, 0.8)
        feature = [1 if random() > p else 0 for _ in range(m)]
        binary_features.append(feature)
    
    continuous_features = [[uniform(-1,1) for _ in range(m)] for _ in range(n_continuous_features)]
    
    features = binary_features + continuous_features
    shuffle(features)
    X = np.array(features).T
    
    y = np.array([np.nan] * m)
    
    def split(nx=None, depth=0):
        nonlocal X,y,k, max_depth
        
        # Index passed into this frame of the function
        if nx is None:
            nx = list(range(len(X)))
            
        # nx must be a list
        if not isinstance(nx, list):
            nx = list(nx)
        
        # If the nx is empty
        if nx == [] or nx == ():
            return
        
        # Decide whether to end this batch into a leaf
        p = random()
        threshold = min(depth / max_depth, len(nx) / min_leaf)
        if p < threshold and (len(nx) < len(X)):
            counts = (tuple(np.bincount(y[~np.isnan(y)].astype(int))) + tuple(range(k)))[:k]
            c = np.argmin(counts) if len(counts) else randint(0, k-1)
            y[nx] = c
            return
        
        # Select a feature
        j = randint(0, len(X[0])-1)
        q = uniform(0.2, 0.8)
        v = np.quantile(X[nx, j], q=q)
        
        # Separet the index into two
        nx_left, nx_right = [], []
        for ix in nx:
            if X[ix, j] >= v:
                nx_left.append(ix)
            else:
                nx_right.append(ix)
        
        # Recurse - left and right
        split(nx_left, depth=depth+1)
        split(nx_right, depth=depth+1)
        
    # Do the split recursively
    split()
    y = tuple(y.astype(int))
    return X,y


###############################################################


if __name__ == '__main__':
    m = 400
    n = 5
    k = 3
    
    X,y = make_data_for_decision_trees(m=400, n=5, k=3,
                                       proportion_of_binary_features=None)
    
    print(y)
    print(k)
    print(np.bincount(y))
    
    
    # TEST
    from sklearn.tree import DecisionTreeClassifier
    md = DecisionTreeClassifier()
    md.fit(X,y)
    acc = md.score(X,y)
    print(acc)
