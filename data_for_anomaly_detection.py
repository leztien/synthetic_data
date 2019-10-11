
"""
function that creates data for anomaly detection
class for Anomaly Detection
"""

import numpy as np


def make_data_for_anomaly_detection(m=10000, 
                                    n=3,
                                    proportion_of_anomalies=0.002,
                                    accentuate_anomalies=False,
                                    shift_data_into_positive=False,
                                    add_feature_transformation=False,
                                    add_redundant_features=False,
                                    add_feature_engeneering=False,
                                    random_state=None):
    if random_state: np.random.seed(random_state)
    for _ in range(100):
        diagonal = np.sort(np.random.randint(1,n*2+n, size=n))
        Σ = np.diag(diagonal)
        mx = diagonal.max()
        a = np.sort(np.random.randint(0, np.floor(mx**0.5), size=((n**2-n)//2)))
        signs = np.array([1,-1])[np.random.randint(0,2, len(a))]
        a *= signs
        how_many = int(np.floor((n**2-n)//2 / 3))
        nx = np.random.permutation(len(a))[:how_many]
        a[nx] = 0
        nx_upper = np.triu_indices(len(Σ),1)
        Σ[nx_upper] = a
        Σ = Σ | Σ.T
        
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("error")
            try: X = np.random.multivariate_normal(mean=[0]*len(Σ), cov=Σ, size=m)
            except(RuntimeWarning): 
                continue
            else: break
    else: X = np.random.multivariate_normal(mean=[0]*len(Σ), cov=Σ, size=m)
    
    
    μ = X.mean(0)
    Σ = np.cov(X.T, ddof=0)
    
      
    def pdf(X, mean, Sigma):
        """functionality of scipy.stats import multivariate_normal.pdf"""
        n = len(mean)
        normalizer = (2*np.pi)**(n/2) * np.sqrt(np.linalg.det(Sigma))
        numerator = np.array([np.exp(-0.5 * x.dot(np.linalg.inv(Sigma)).dot(x)) for x in (X-mean)])
        densities = numerator / normalizer
        return densities
    
    
    from scipy.stats import multivariate_normal
    mvg = multivariate_normal(mean=μ, cov=Σ)
    densities = mvg.pdf(X)
    
    n_anomalous = max(1, int(proportion_of_anomalies*m))  # 0.2% are anomalies
    nx = np.argpartition(densities, kth=n_anomalous)[:(n_anomalous)]
    y = np.zeros(shape=m, dtype='uint8')
    y[nx] = 1
    
    if accentuate_anomalies:
        nx = nx[::2]
        default_anomaly_accenuating_coefficient = 0.5
        anomaly_accenuating_coefficient = accentuate_anomalies if(isinstance(accentuate_anomalies,float)) else default_anomaly_accenuating_coefficient
        error_multipliers = (np.random.normal(0,anomaly_accenuating_coefficient,size=X[nx].shape).__abs__()+1)
        how_many = error_multipliers.size - max(1, int(np.sqrt(n).round(0))) * len(error_multipliers)
        nx2 = np.random.permutation(error_multipliers.size)[:how_many]
        error_multipliers.ravel()[nx2] = 1
        X[nx] *= error_multipliers
    
    if shift_data_into_positive:
        X -= X.min(0)
    
    if add_feature_transformation:
        pass
    
    if add_redundant_features:
        pass
    
    if add_feature_engeneering:
        pass
    
    return(X,y)

#===============================================================================
    
class AnomalyDetection:
    def __init__(self, multivariate = False, criterion = "f1_score"):
        self.multivariate = multivariate
        self.criterion = criterion
        
        
    def fit(self, X, y=None):
        from scipy.stats import norm, multivariate_normal
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        mask = np.equal(y,1)
        m0,m1 = np.bincount(mask)
        
        Xtr = X[~mask][:int(m0*.6)]

        Xcv = np.concatenate([X[~mask][int(m0*.6):int(m0*.8)], X[mask][:int(m1*.5)]], axis=0)
        ycv = np.concatenate([y[~mask][int(m0*.6):int(m0*.8)], y[mask][:int(m1*.5)]])
        
        self.Xts = np.concatenate([X[~mask][int(m0*.8):], X[mask][int(m1*.5):]], axis=0)
        self.yts = np.concatenate([y[~mask][int(m0*.8):], y[mask][int(m1*.5):]])
        
        #fitting process
        mu = Xtr.mean(0)
        
        if self.multivariate:
            Σ = np.cov(Xtr.T)
            distribution = multivariate_normal(mean=mu, cov=Σ)
            self.predict_densities = lambda X : distribution.pdf(X)
        
        else:
            std = np.std(Xtr, ddof=0)
            distribution = norm(loc=mu, scale=std)
            self.predict_densities = lambda X : np.add.reduce(np.log(distribution.pdf(X)), axis=1)
        
        densities = self.predict_densities(Xcv)
        epsilons_sorted = sorted(densities)
        ix = sum(ycv==1)
        
        for ix in range(ix,len(epsilons_sorted)):
            ypred = densities < epsilons_sorted[ix]
            recall = recall_score(ycv,ypred)
            if recall >= 1.0: 
                epsilon_lower = epsilons_sorted[1]
                epsilon_upper = epsilons_sorted[ix+25]  # 25 is arbitrary number 
                break
        
        epsilons = np.linspace(epsilon_lower, epsilon_upper, 100)
        recalls, precisions, f1_scores = (list() for _ in range(3))
        
        for epsilon in epsilons:
            ypred = densities < epsilon
            recall = recall_score(ycv, ypred)
            precision = precision_score(ycv, ypred)
            f1 = f1_score(ycv, ypred)
            [l.append(v) for l,v in zip([recalls, precisions, f1_scores],[recall, precision, f1])]
        
        #make pandas-df
        from pandas import DataFrame, set_option
        set_option('precision', 2, 'display.width', 250)
        
        d = {"recall":recalls, "precision":precisions, "f1_score":f1_scores, "epsilon":epsilons}
        df = DataFrame(d)
        self.df = df.groupby(by=["recall","precision","f1_score"]).mean().reset_index().sort_values("epsilon").reset_index(drop=True)
        print(self.df)
        
        ix = df.idxmax()[self.criterion]
        self.optimal_epsilon = df.iloc[ix]["epsilon"]
        return(self)
    
    
    def test(self, X=None, y=None):
        X,y = (X or self.Xts), (y or self.yts)
        densities = self.predict_densities(X)
        ypred = densities < self.optimal_epsilon
        from sklearn.metrics import classification_report
        s = classification_report(y, ypred)
        print(s)
        return(self)
    
    
    def predict(self, X):
        densities = self.predict_densities(X)
        ypred = densities < self.optimal_epsilon
        return ypred

#==================================================================================


X,y = make_data_for_anomaly_detection(m=10000, n=100, 
                                      accentuate_anomalies=True, 
                                      shift_data_into_positive=True,
                                      random_state=None)


md = AnomalyDetection(multivariate=True, criterion="recall").fit(X,y)
md.test()
ypred = md.predict(X)

