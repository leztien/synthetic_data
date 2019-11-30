
import numpy as np
def make_data_for_ANN(m=100, n=3, K=2, L=1, u=16, gmm=False, balanced_classes=True, seed=None):
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
    K = number of classes
    L = number of hidden layers
    u = number of units in the hidden layers
    gmm = fit GMM's to the generated data and draw samples from these GMM's (if int = number of gaussians per class)
    balanced_classes = average entropy to ensure (relatively) balanced classes (False = only ensure presence of all K classes)
    seed = if True then the generated seed is printed out for the user's reference
    """

    #error checking for the arguments
    assert balanced_classes in (False,None,True) or 0 < balanced_classes < 1.0,"error1"
    if balanced_classes: balanced_classes = 0.98 if balanced_classes is True else balanced_classes  #default value
    assert n >= 2,"error2"
    assert all(isinstance(e,int)and(e>0) for e in (m,n,K,L,u)),"error3"

    #randomness
    if seed:
        if seed is True:
            from random import randint
            seed = randint(0,int(2**32-1))
            print("random seed =", seed, "(you can use this random number to reproduce the generated dataset)")
            make_data_for_ANN.seed = seed
        np.random.seed(seed)

    #utility functions
    def avg_entropy(y): # measures the balance of classes (the higher the better)
        counts = np.bincount(y)
        p = counts / sum(counts)
        return -np.dot(p, np.log2(p)) / np.log2(len(p))

    def softmax(Z):  #activation function for the output layer
        return np.exp(Z) / np.exp(Z).sum(0, keepdims=True)

    #create data
    X = np.random.uniform(-1,1, size=(m,n))

    #find suitabel weights
    shapes = (n,) + (u,)*L + (K,)  #e.g. (3, 32, 32, 2)
    temp = (0, 'y')
    for attempt in range(300):  # number of attempts to produce a balanced dataset
        g = zip(shapes[1:], shapes[:-1])  #shape-tuples
        WW = [np.random.normal(loc=0, scale=1, size=shape) for shape in g]

        #forward propagation
        A = X.T
        for l,W in enumerate(WW):
            Z = np.matmul(W,A)
            A = (np.tanh if l<len(WW)-1 else softmax)(Z)
        y = A.T.argmax(1)

        #check presence of all classes and class balance:
        if len(set(y)) == K: #ensures all classes are present (although maybe unbalanced)
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
        if isinstance(y, str) or len(set(y))<K:
            class MissingClassesGenerated(BaseException):pass
            msg = "Generated labels miss a class or classes. Increase the number of units/layers"
            raise MissingClassesGenerated(msg)

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
    X,y = make_data_for_ANN(m=10000, n=2, K=5, L=3, u=16, gmm=True, balanced_classes=True, seed=True)

    m,n = X.shape
    print("labels counts =", np.bincount(y))

    from sklearn.neural_network import MLPClassifier
    md = MLPClassifier((16,16,16))
    md.fit(X,y)
    acc = md.score(X,y)
    print("sklearn MLP accuracy =", round(acc,2))

    if n == 2:
        import matplotlib.pyplot as plt
        plt.scatter(*X.T, c=y, s=5, cmap='rainbow')
        plt.show()
