import numpy as np

def cluster_decoding(X,Y,T,K,cluster_method = 'regression',\
    cluster_measure = 'error',Pstructure = None,Pistructure = None,\
    GammaInit = [], repetitions =100, nwin = 0):
    """
    clustering of the time-point-by-time-point regressions, which is
    temporally constrained unlike TUDA
    INPUT
        X,Y,T are as usual
        K is the number of states
        cluster_method is 'regression', 'hierarchical', or 'sequential'
        cluster_measure is 'error', 'response' or 'beta', only used if cluster_method is 'hierarchical'
        Pstructure and Pistructure are constraints in the transitions
        GammaInit: Initial state time course (optional)
        repetitions: How many times to repeat the init (only used if cluster_method is 'sequential'
    OUTPUT
        Gamma: (trial time by K), containing the cluster assignments
    """
    N = np.shape(T); p = np.shape(X)[1]; q = np.shape(Y)[1]; ttrial = T[0]

    if Pstructure == None : Pstructure = np.ones((K,1), dtype=bool)
    if Pistructure == None : Pistructure = np.ones(K, dtype=bool)
    if nwin == 0 :
        swin = 1
    else :
        nwin = min(50,ttrial)
        swin = int(ttrial/nwin)

