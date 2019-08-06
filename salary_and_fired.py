
"""
function that makes a dataset:
    dataset "salary": dataset for regression. predict salary from age, years of experience, education level and number of subordinates
    dataset "classification": predict whether an employee was fired
"""

import numpy as np
import matplotlib.pyplot as plt


def adjust_m(feature, m):
    a = feature if isinstance(feature, list) else list(feature)
    from random import shuffle
    from numpy import array
    shuffle(a)
    if len(a) < m:
        l = a[:m-len(a)]
        a = a + l
    elif len(a) > m:
        a = a[:m]
    else:pass
    return array(a)


def make_data(m=100, dataset="salary"):
    #x1 = Age in whole years
    from statsmodels.sandbox.distributions import extras
    pdf = extras.pdf_mvsk
    pdf = pdf((40, 10**2, 1, -0.75))
    xx = np.arange(16,90)
    yy = pdf(xx)
    yy = yy / yy.sum()
    a = sum(([x,] * int(round(y*m)) for x,y in zip(xx,yy)),[])
    x1 = adjust_m(a, m).astype('uint8')
    
    #x2 = Work Experience in years
    a = np.clip(x1 - 16, 0, 60)
    deviations = np.random.normal(0, scale=2, size=m)
    x2 = a + deviations
    x2 = np.clip(x2, 0, x1-16).round().astype(np.uint8)
    assert np.all(x2 <= x1-16),"error"
    
    #x3 = Education (school, colledge, degree, higher)
    from scipy.stats import binom
    n,p = 3, 0.45
    distribution = binom(n,p)
    a = distribution.rvs(size=m) + 1
    x3 = adjust_m(a, m).astype('uint8')
    
    #x4 = number of subordinates
    from scipy.stats import chi2
    distribution = chi2(df=1, loc=0, scale=4)
    a = distribution.rvs(size=m).astype(int)
    x4 = adjust_m(a, m)
    r = x2 / x2.max()
    x4 = ((r*x4) ** 1.75).round().astype(np.uint16)
    
    #target
    #education gives us 4 starting biases
    b3 = x3 * 1000  #education level multiplyied by 1000
    f2 = x2 * (b3/20)  #work experience: starimg salary * years of experience
    f4 = np.log(x4+1) * 1000  #number of subordinates
    
    # age dependence: parabolic relatonship
    f1 =  3 - 0.005 * (x1-45)**2   #age - parabolic curve
    r = 0.0        # ratio of dependence
    y = (b3 + f2 + f4) * (1 + r*f1)  * 12  # slary per annum
    y = y / 1.5
    y = (y//250*250).astype('float32')
    
    from pandas import DataFrame
    columns = ["Age", "Work Experience", "Education Level", "Number of Subordinates", "Salary"]
    d = {name:feature for name,feature in zip(columns, [x1,x2,x3,x4,y])}
    df = DataFrame(d)
    if str(dataset).lower() in ("salary","regression"): return df
    
    #continue making the logistic regression data-set
    salary_soll = y
    salary_ist = y + np.random.normal(loc=0, scale=y/10, size=m)
    df["Salary"] = (salary_ist//250*250).astype('float32')
    
    #claculations of risk
    ratio = salary_ist / salary_soll
    z = ratio * 100
    z = z - z.min()
    age_risk = x1.astype(np.float16)**2 
    age_risk = age_risk - age_risk.min()
    age_risk = age_risk / (age_risk.max() / z.max())
    risk = z + age_risk*0.25
    mask = risk > np.percentile(risk, 99)
    df["Fired"] = mask.astype(np.uint8)
    return df
        

#===============================================================


df = make_data(m=10000, dataset="salary")
df.to_csv("data_for_regression.csv", index=False)
X = df.iloc[:,:-1].values
y = df["Salary"].values


df = make_data(m=10000, dataset="classification")
df.to_csv("data_for_classification.csv", index=False)
X = df.iloc[:,:-1].values
y = df["Fired"].values

