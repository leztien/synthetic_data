
"""
demo of a multi-output classifier
"""

import numpy as np
import matplotlib.pyplot as plt


def make_data():
    m,n = (1000, 2)
    mx = np.random.uniform(-5,5, size=(m,n))
    
    mask = np.square(mx).__mul__([1,2]).sum(1) < (5**2)
    X = mx[mask]
    
    y1 = (np.square(X).sum(1) < (3**2)).astype('uint8')
    
    def get_tri_labels(X):
        func = lambda x1,x2 : x2 > x1 if x1 < 0 else x2 > -x1
        y = [func(x1,x2) for x1,x2 in X]
        for i,x1 in enumerate(X[:,0]):
            if y[i] and x1>0: y[i] = 2
        return y
    
    y2 = get_tri_labels(X)
    Y = np.c_[y1,y2]
    return(X,Y)

#===============================================================

X,Y = make_data()


#visualize the data
for label1 in sorted(set(e for e in Y[:,1])):
    mask1 = Y[:,1]==label1
    for label2 in sorted(set(e for e in Y[:,0])):
        mask2 = Y[:,0]==label2
        plt.plot(*X[mask1 & mask2].T, color='rgb'[label1], marker="o."[label2], linestyle='none')



#KNN (100%)
from sklearn.neighbors import KNeighborsClassifier
md = KNeighborsClassifier(n_neighbors=3, weights='uniform').fit(X,Y)
Ypred = md.predict(X)

cc = sorted(set(tuple(e) for e in Y))
ytrue = [cc.index(tuple(e)) for e in Y]
ypred = [cc.index(tuple(e)) for e in Ypred]
accuracy = sum(y1==y2 for y1,y2 in zip(ytrue,ypred)) / len(ytrue)
print(accuracy)

#evaluate
from sklearn.metrics import classification_report, f1_score, recall_score
s = classification_report(ytrue, ypred)
print(s)

f1 = f1_score(ytrue, ypred, average='macro')
print("f1-score (macro) =", f1.round(3))

recall = recall_score(ytrue, ypred, average='macro')
print("recall (macro) =", recall.round(3), end="\n\n")


#Gaussian (multiputput)
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
est = GaussianNB()
md = MultiOutputClassifier(est)
md.fit(X,Y)

Ypred = md.predict(X)
accuracy = md.score(X,Y)
print("Gaussian Naive Bayes multioutput accuracy =", accuracy.round(2))


#Logistic (multiputput)
from sklearn.svm import SVC
est = SVC(gamma='auto')
md = MultiOutputClassifier(est)
md.fit(X,Y)

Ypred = md.predict(X)
accuracy = (Y==Ypred).all(1).mean()  # this si how multiput accuracy is calculated
print("SVM multioutput accuracy =", accuracy.round(2))
