"""
make data for a decsion tree regressor
half of the features are continuous, and half is categoirical (including binary)
"""


from random import random, uniform, shuffle, randint, choice
import numpy as np


def make_data_for_decision_tree_regressor(m, n, categorical_features_proportion=0.5):
    # Arbitrary values for max_depth and min_points_in_a_leaf
    k = randint(2, 11)
    max_depth = np.log(m) * n * k / 10
    min_leaf = np.log(m) * n * k / 25
    
    n_categorical_features = round(n * categorical_features_proportion)
    n_continuous_features = n - n_categorical_features
    
    categorical_features = list()
    for j in range(n_categorical_features):
        if random() < 0.5:
            p = uniform(0.2, 0.8)
            feature = [1 if random() > p else 0 for _ in range(m)]
        else:
            r = range(randint(3, max(m//10, 4)))
            feature = [choice(r) for _ in range(m)]
        categorical_features.append(feature)
            
    
    continuous_features = [[uniform(-1,1) for _ in range(m)] for _ in range(n_continuous_features)]
    
    features = categorical_features + continuous_features
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
    y = y.astype(int)
    
    # patch to return continuous values
    p = randint(3, 9)
    X_poly = [X**p for p in range(1,p)]
    X_poly = np.hstack(X_poly)
    X_st = (X_poly - X_poly.mean(axis=0)) / X_poly.std(axis=0)
    weights = np.random.normal(size=X_st.shape[1])
    y_cont = (X_st * weights).sum(axis=1)
    y = y_cont + (y - y.mean())
    y = y - y.min()
    return X,y




if __name__ == '__main__':
    m = choice([100,500,1000])
    n = randint(1,10)
    max_depth = randint(3, int(np.log(m*n)))
    
    
    X,y = make_data_for_decision_tree_regressor(m, n, categorical_features_proportion=0.5)
    
    
    print(f"m = {m}\tn = {n}\ntree max depth = {max_depth}")
    
    from sklearn.tree import DecisionTreeRegressor
    md = DecisionTreeRegressor(max_depth=max_depth).fit(X,y)
    rsq = md.score(X,y)
    print("sklearn accuracy:", rsq.round(2))
    
    
    
    

