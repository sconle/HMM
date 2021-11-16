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

    def smooth(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    N = np.shape(T); p = np.shape(X)[1]; q = np.shape(Y)[1]; ttrial = T[0]

    if Pstructure == None : Pstructure = np.ones((K,1), dtype=bool)
    if Pistructure == None : Pistructure = np.ones(K, dtype=bool)
    if nwin == 0 :
        swin = 1
    else :
        nwin = min(50,ttrial)
        swin = int(ttrial/nwin)

    to_use = np.ones((ttrial,1),dtype=bool)


    if swin > 1:
        r = np.remainder(ttrial,nwin)
        if r > 0:
            to_use[:-r] = False


    X = np.reshape(X,[ttrial, N, p])
    Y = np.reshape(Y,[ttrial, N, q])

    if swin > 1 :
        X = X[to_use,:,:]
        X = np.reshape(X,[swin, nwin, N, p])
        X = np.transpose(X,[2, 1, 3, 4])
        X = np.reshape(X,[nwin, N*swin, p])
        Y = Y[to_use,:,:]
        Y = np.reshape(Y,[swin, nwin, N, q])
        Y = np.transpose(Y,[2, 1, 3, 4])
        Y = np.reshape(Y,[nwin, N*swin, q])
        ttrial0 = ttrial; N0 = N
        ttrial = nwin; N = N*swin; T = nwin * np.ones((N,1))

        if cluster_method=='regression':
            max_cyc = 100; reg_parameter = 1e-5; smooth_parameter = 1
            # start with no constraints
            if GammaInit == []:
                Gamma = cluster_decoding(np.reshape(X,[ttrial*N, p]),np.reshape(Y,[ttrial*N, q]),\
                    T,K,'sequential',[],[],[],[],10,1)
            else:
                Gamma = GammaInit; 

            assig = np.zeros((ttrial,1))
            for t in range(ttrial):
                assig[t] = np.nonzero(Gamma[t,:]==1)
            for t in range(ttrial):
                assig(t) = np.nonzero(Gamma[t,:]==1)
            j1 = assig[0]
            if ~Pistructure(j1): # is it consistent with constraint?
                j = np.nonzero(Pistructure,1)
                Gamma_j = Gamma[:,j]
                Gamma[:,j] = Gamma[:,j1]
                Gamma[:,j1] = Gamma_j
                for t in range(ttrial):
                     assig(t) = np.nonzero(Gamma[t,:]==1)

            assig_pr = assig
            beta = np.zeros((p,q,K))
            err = np.zeros((ttrial,K))
            for cyc in range(max_cyc):
                # M
                for k in range(K):
                    ind = assig==k
                    Xstar = np.reshape(X[ind,:,:],[sum(ind)*N, p])
                    Ystar = np.reshape(Y[ind,:,:],[sum(ind)*N, q])

                    #### a modif avec des @
                    beta[:,:,k] = (Xstar.T * Xstar + reg_parameter * np.eye(size(Xstar,2)))*(Xstar.T * Ystar)^(-1)

                # E
                Y = np.reshape(Y,[ttrial*N, q])
                for k in range(K):
                    Yhat = np.reshape(X,[ttrial*N, p]) * beta[:,:,k]
                    e = np.sum(np.pow((Y - Yhat),2),2)
                    e = np.reshape(e,[ttrial, N])
                    err[:,k] = np.sum(e,2)
                    err[:,k] = smooth(err[:,k],smooth_parameter)

                Y = np.reshape(Y,[ttrial, N, q])
                err(1,~Pistructure) = float('inf') 
                assig(1) = np.argmin(err[1,:]) 
                for t in range(1,ttrial):
                    err[t,~Pstructure[assig[t-1],:]] = np.Infinity
                    assig(t) = np.argmin(err[t,:])

                # terminate?
                #if ~all(Pstructure(:)), keyboard; end
                if all(assig_pr==assig):
                    break
                assig_pr = assig
            for t in range(ttrial):
                Gamma[t,:] = 0
                Gamma[t,assig(t)] = 1