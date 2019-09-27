import numpy as np

def make_data(m=100, n=3, n_classes=2, t:'number of trials'=10):
    mm = np.random.random(n_classes)
    mm = [int(m*p) for p in mm/sum(mm)]
    
    mm_sum = sum(mm)
    if mm_sum<m:
        ix = mm.index(min(mm))
        mm[ix] += (m-mm_sum)
    elif mm_sum > m:
        ix = mm.index(max(mm))
        mm[ix] -= (mm_sum-m)
        
    pp = [np.random.random(n+1) for _ in mm]
    
    for i,p in enumerate(pp):
        l = list(pp[i])
        l.append(l.pop(l.index(max(l))))
        pp[i] = l
        
    pp = [pp/sum(pp) for pp in pp]
    
    XX = [np.random.multinomial(n=t, pvals=p, size=m) for m,p in zip(mm,pp)]
    X = np.concatenate(XX, axis=0)[:,:-1]
    y = sum([[c]*m for m,c in zip(mm, range(n_classes))], [])
    return X,y

#=================================================================================

X,y = make_data(m=100, n=4, n_classes=3, t=10)
