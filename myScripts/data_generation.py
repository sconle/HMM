# In this .py script, I implement a function to generate data from the cluster_decoder class
# @author: valentin

### import librairies ###
print(f"Importing librairies ... ", end='')
import sys
sys.path.append(r"C:\Users\valentin\Documents\HMM")
import numpy as np
from myHmmPackage.cluster_decoder import ClusterDecoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
print('done')

### data ###
n_samples = 100
n_time_points = 100
n_regions = 2
n_label_features = 48
n_models = 4

y = (np.random.randint(100, size=(n_samples, n_time_points, n_regions)) == 0).astype(int)
eps = np.random.normal(0,10, size=(n_samples, n_time_points, n_regions))
X = np.random.randint(100,size=(n_samples, n_time_points, n_label_features))

W = np.random.randint(low=-10, high=10,size=(n_models, n_label_features, n_regions))
W_ = np.reshape(W, (n_label_features, n_models, n_regions))
gamma = np.random.normal(0,1, size=(n_models))
y = X @ (gamma @ W_) + eps
y = np.reshape(y, newshape=(n_samples, n_time_points, n_regions))

if __name__=='__main__':
    print(f"Starting cluster decoder ... ")
    clf = ClusterDecoder(n_clusters=n_models, method='hierarchical', max_iter=1e3)
    clf.fit(X,y) 
    print('done\n')

    W_true_flatten = W.flatten()
    W_predict_flatten = clf.decoding_mats_.flatten()
    print(f"r2_score : {r2_score(W_true_flatten, W_predict_flatten)}")
    print(f"mean_squared_error : {mean_squared_error(W_true_flatten, W_predict_flatten)}")    
    print(f"mean_absolute_percentage_error : {mean_absolute_percentage_error(W_true_flatten, W_predict_flatten)}")