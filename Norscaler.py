# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:11:57 2022

@author: Administrator
"""

def Norscaler(Train, Test):
    X= np.vstack([Train, Test])
    s = StandardScaler()
    s.fit(X)
    flatTrain = s.transform(Train)
    flatTest = s.transform(Test)
    return flatTrain, flatTest