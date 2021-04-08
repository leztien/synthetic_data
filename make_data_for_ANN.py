import numpy as np
def make_data_for_ANN(m=100, n=3, k=2, l=1, u=16, 
                      balanced_classes=True, space_between_classes=False, 
                      gmm=False, seed=None):
    """
    function generating synthetic multidimensional data for ANN by forward propagation
    -the multidimensional space is linearly filled with data-points
    -the weights-matreces are filled with values from normal distribution
    -the data is forward-propagated through the weights
    -optionally (spherical) GMM's can be fit to the generated data and data-points sampled from these GMM's
    -the data-points are in the range of (-1,+1)
    -the data is sorted by the labels (i.e. rememebr to shuffle!)
    m = number of data-points
    n = number of features
    k = number of classes
    l = number of hidden layers
    u = number of units in the hidden layers
    
    balanced_classes = average entropy to ensure (relatively) balanced classes (False = only ensure presence of all K classes)
    space_between_classes = space between classes is achieved by deleting points with lower probabilities (higher value = more space)
    gmm = fit GMM's to the generated data and draw samples from these GMM's (if int = number of gaussians per class)
    seed = if True then the generated seed is printed out for the user's reference
    """

    #error checking for the arguments
    assert balanced_classes in (False,None,True) or 0 < balanced_classes < 1.0,"error1"
    if balanced_classes: balanced_classes = 0.98 if balanced_classes is True else balanced_classes  #default value
    assert n >= 2,"error2"
    assert all(isinstance(e,int)and(e>0) for e in (m,n,k,l,u)),"error3"
    #take care of variable space_between_classes:
    assert space_between_classes in (False,None,True) or 0 < space_between_classes < .7,"error4"
    if space_between_classes:
        space_between_classes = 0.25 if space_between_classes is True else space_between_classes  #default value
        m_original = m
        m = int(m / (1-space_between_classes))

    #randomness
    if seed:
        if seed is True:
            from random import randint
            seed = randint(0, 1E4)  # int(2**32-1)
            print("random seed =", seed)
            make_data_for_ANN.seed = seed
        np.random.seed(seed)

    #utility functions
    def avg_entropy(y): # measures the balance of classes (the higher the better)
        counts = np.bincount(y)
        p = counts / sum(counts)
        return -np.dot(p, np.log2(p)) / np.log2(len(p))

    def softmax(Z):  #activation function for the output layer
        return np.exp(Z) / np.exp(Z).sum(0, keepdims=True)

    #CREATE DATA
    X = np.random.uniform(-1,1, size=(m,n))

    #GENERATE APPROPRIATE WEIGHTS (in a loop)
    shapes = (n,) + (u,)*l + (k,)  #e.g. (3, 32, 32, 2)
    shapes = tuple(zip(shapes[1:], shapes[:-1]))  #shape-tuples
    temp = (0, 'y')
    
    for attempt in range(500):  # number of attempts to produce a balanced dataset
        WW = [np.random.normal(loc=np.random.uniform(-0.5, 0.5), scale=5, size=shape) for shape in shapes]
        bb = [np.random.normal(loc=np.random.uniform(-1, 1), scale=np.random.uniform(0,5), size=shape[0])[:,None] for shape in shapes]
        
        #forward propagation
        A = X.T
        for l,(W,b) in enumerate(zip(WW,bb)):
            Z = np.matmul(W,A) + b
            A = (np.tanh if l<len(WW)-1 else softmax)(Z)
        P = A.T
        y = A.T.argmax(1)
        
        #check presence of all classes and class balance:
        if len(set(y)) == k: #ensures all classes are present (although maybe unbalanced)
            if not balanced_classes: break
            H = avg_entropy(y)
            if H >= balanced_classes: break
            elif H > temp[0]: temp = (H, y)
    else: # if after all the loops failed to generate balanced classes
        from warnings import warn
        msg = "Failed to produce balanced numbers of class-labels. Increase the number of units/layers"
        warn(msg, Warning)
        y = temp[1]  # get the labels with the best yet average entropy
        #check the y
        if isinstance(y, str) or len(set(y)) < k:
            class MissingClasses(BaseException):pass
            msg = "Generated labels miss a class or classes. Increase the number of units/layers"
            raise MissingClasses(msg)
    #-----end of the main loop----------------
    
    #add space between classes
    if space_between_classes:
        threshold = np.quantile(P.max(1), q=space_between_classes)
        mask = P.max(1) > threshold
        X = X[mask]  # the points with lower probabilities are discarded
        y = y[mask]  # the points with lower probabilities are discarded
        m = len(y)   # do not delete this!
        
        #add or remove data-points to ammount to the original m
        if len(y) < m_original:
            c = np.bincount(y).argmin() # c = the class with the loweest number of data-poinst
            n_missing = m_original - len(y)
            mask = y==c
            nx = np.random.permutation(len(X[mask]))[:n_missing]
            
            yadditional = y[mask][nx]
            Xadditional = X[mask][nx] + np.random.normal(0, 0.001, size=(len(yadditional),n))
            
            X = np.vstack([X,Xadditional])
            y = np.concatenate([y,yadditional])
            m = len(y)    #do not delete this!
            if len(set(y)) < k:
                from warnings import warn
                msg = "missing class(es)! Turn on 'balanced_classes', deactivate 'space_between_classes' or increase m"
                warn(msg, Warning)
                
        elif len(y) > m_original:
            raise RuntimeWarning("ADD CODE FOR: too many data-points")
    
    #GMM
    if gmm:
        from sklearn.mixture import GaussianMixture
        default_n_gaussians = 25
        n_gaussians = ((abs(int(gmm))-1) or default_n_gaussians-1)+1
        classes = sorted(set(y))
        MD = [GaussianMixture(n_components=n_gaussians, covariance_type='spherical').fit(X[y==c]) for c in classes]
        mm = [X[y==c].shape[0] for c in classes]
        y = sum([[k]*m for k,m in zip(range(len(classes)),mm)], [])
        X = np.vstack([md.sample(m)[0] for md,m in zip(MD,mm)])

    #relabel the targets (so that their class numbers are in ascending order)
    nx = np.bincount(y).argsort()
    d = {k:v for k,v in zip(nx, range(len(nx)))}
    y = np.array([d[k] for k in y], dtype='uint8')
    nx = np.argsort(y)
    X,y = (nd[nx] for nd in (X,y))
    return(X,y)

###########################################################################################


if __name__ == '__main__':  #DEMO
    X,y = make_data_for_ANN(m=10000, n=2, k=5, l=2, u=8, 
                            balanced_classes=True, space_between_classes=True, 
                            gmm=True, seed=True)

    m,n = X.shape
    print("labels counts =", np.bincount(y), m)
    
    
    from sklearn.neural_network import MLPClassifier
    md = MLPClassifier((16,16,16))
    md.fit(X,y)
    acc = md.score(X,y)
    print("sklearn MLP accuracy =", round(acc,2))
    
    
    if n == 2:
        import matplotlib.pyplot as plt
        plt.scatter(*X.T, c=y, s=5, cmap='rainbow')
        plt.show()

