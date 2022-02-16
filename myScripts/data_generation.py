# In this .py script, I implement a function to generate data from the cluster_decoder class
# @author: valentin

### import librairies ###
import numpy as np
from myHmmPackage.cluster_decoder import *

### data ###
y = (np.random.randint(100, size=1000) == 0).astype(int)
eps = np.random.normal(0,10, size=1000)
W = np.random.randint(low=-10, high=10,size=(1000,1000))
X = (y -eps) @ np.inv(W)
if __name__=='__main__':
    clf = ClusterDecoder()
    clf.fit(X,y)