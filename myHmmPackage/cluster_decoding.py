import numpy as np
from numpy.lib.function_base import insert
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import to_tree
from scipy.cluster.hierarchy import fcluster


def cluster_decoding(X, Y, T, K, cluster_method='regression',\
    cluster_measure = 'error', Pstructure = None, Pistructure = None,\
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
    def inspect(var):
        print(var)
        print(np.shape(var))
        print('yoooooo')
####### Début Initialisation #######

    def smooth(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    N = np.shape(T)[0]; p = np.shape(X)[1]; q = np.shape(Y)[1]; ttrial = int(T[0])

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
            to_use[-r:] = False 


    X = np.reshape(X,[ttrial, N, p]) 
    Y = np.reshape(Y,[ttrial, N, q])

    if swin > 1 :
        X = [X[i,:,:] for i in range(len(to_use)) if to_use[i]]
        X = np.reshape(X,[swin, nwin, N, p])
        X = np.transpose(X,[1, 0, 2, 3])
        X = np.reshape(X,[nwin, N*swin, p])

        Y = [Y[i,:,:] for i in range(len(to_use)) if to_use[i]]
        Y = np.reshape(Y,[swin, nwin, N, q])
        Y = np.transpose(Y,[1, 0, 2, 3])
        Y = np.reshape(Y,[nwin, N*swin, q])
        ttrial0 = ttrial; N0 = N
        ttrial = nwin; N = N*swin; T = nwin * np.ones((N,1))

####### Fin Initialisation #######

####### Début Methode Régression #######

    if cluster_method=='regression':
        max_cyc = 100; reg_parameter = 1e-5; smooth_parameter = 1
        # start with no constraints
        if GammaInit == []:
            Gamma = cluster_decoding(np.reshape(X,[ttrial*N, p]),np.reshape(Y,[ttrial*N, q]),\
                T,K,'sequential',[],[],[],[],10,1)
        else:
            Gamma = GammaInit

        assig = np.zeros((ttrial,1))
        for t in range(ttrial):
            assig[t] = np.nonzero([1 if g==1 else 0 for g in Gamma[t,:]])
        j1 = assig[0]
        if not Pistructure(j1): # is it consistent with constraint?
            j = np.nonzero(Pistructure,1)
            Gamma_j = Gamma[:,j]
            Gamma[:,j] = Gamma[:,j1]
            Gamma[:,j1] = Gamma_j
            for t in range(ttrial):
                 assig[t] = np.nonzero([1 if g==1 else 0 for g in Gamma[t,:]])

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
                beta[:,:,k] = (Xstar.T  @ Xstar + reg_parameter * np.eye(np.size(Xstar,2)))*(Xstar.T * Ystar)^(-1)

            # E
            Y = np.reshape(Y,[ttrial*N, q])
            for k in range(K):
                Yhat = np.reshape(X,[ttrial*N, p]) * beta[:,:,k]
                e = np.sum(np.pow((Y - Yhat),2),2)
                e = np.reshape(e,[ttrial, N])
                err[:,k] = np.sum(e,2)
                err[:,k] = smooth(err[:,k],smooth_parameter)

            Y = np.reshape(Y,[ttrial, N, q])
            #err[1, not Pistructure] = float('inf')
            err[1,:] = [float('inf') if not p else None for p in Pistructure]

            assig[1] = np.argmin(err[1,:])
            for t in range(1,ttrial):
                err[t,:] = [float('inf') if not p else None for p in Pstructure[assig[t-1],:]]
                assig[t] = np.argmin(err[t,:])

            # terminate?
            #if ~all(Pstructure(:)), keyboard; end
            if all(assig_pr==assig):
                break
            assig_pr = assig
        for t in range(ttrial):
            Gamma[t,:] = 0
            Gamma[t,assig(t)] = 1

####### Fin Methode Régression #######

####### Début Methode Hierarchical #######

    elif cluster_method == "hierarchical":
        beta = np.zeros((p, q), ttrial)

        for t in range(ttrial):
            Xt = np.transpose(X[t, :, :], (1, 2, 0))
            Yt = np.transpose(Y[t, :, :], (1, 2, 0))
            beta[:, :, t] = (np.transpose(Xt) @ Xt) @ np.invert(np.transpose(Xt) @ Yt)

        if cluster_measure == "response":
            dist = np.zeros((ttrial * (ttrial - 1) / 2, 1))
            dist2 = np.zeros((ttrial, ttrial))
            Xstar = np.reshape(X, [ttrial * N, p])  # Paranthèse ou non ?
            c = 1

            for t2 in range(ttrial-1):
                d2 = Xstar * beta[:, :, t2]
                for t1 in range(t2+1, ttrial):
                    d1 = Xstar * beta[:, :, t1]
                    dist[c] = np.sqrt(sum(sum((d1 - d2) ** 2)))
                    dist2[t1, t2] = dist[c]
                    dist2[t2, t1] = dist[c]
                    c += 1

        elif cluster_measure == "error":
            dist = np.zeros((ttrial * (ttrial - 1) / 2, 1))
            dist2 = np.zeros((ttrial, ttrial))
            c = 1

            for t2 in range(ttrial-1):
                Xt2 = np.transpose(X[t2, :, :], (1, 2, 0))
                Yt2 = np.transpose(Y[t2, :, :], (1, 2, 0))
                for t1 in range(t2+1, ttrial):
                    Xt1 = np.transpose(X[t1, :, :], (1, 2, 0))
                    Yt1 = np.transpose(Y[t1, :, :], (1, 2, 0))
                    error1 = np.sqrt(sum(sum((Xt1 * beta[:, :, t2] - Yt1) ** 2)))
                    error2 = np.sqrt(sum(sum((Xt2 * beta[:, :, t1] - Yt2) ** 2)))
                    dist[c] = error1 + error2
                    c += 1
                    dist2[t1, t2] = error1 + error2
                    dist[t2, t1] = error1 + error2

        elif cluster_measure == "beta":
            beta = np.transpose(beta, [2, 0, 1])
            beta = np.reshape(beta, [ttrial, p * q])
            dist = distance.pdist(beta)

        if distance.is_valid_dm(np.transpose(dist)): # A checker
            link = to_tree(linkage(np.transpose(dist), "ward"))  # A checker
        else:
            link = to_tree(linkage(np.transpose(dist))) # A checker

        assig = fcluster(link, criterion="maxclust", R=K)  # A checker

####### Fin Methode Hierarchical #######

####### Début Methode Sequential #######

    elif cluster_method == "sequential":
        regularization = 1.0
        assig = np.zeros((ttrial, 1))
        err = 0
        changes = [1] + [int(i * np.round(ttrial / K)) for i in range(1, K)] + [ttrial]        
        Ystar = np.reshape(Y, [ttrial * N, q])

        for k in range(1, K + 1):
            assig[changes[k]:changes[k+1]] = k
            ind = assig == k
            Xstar = np.reshape([X[i,:,:] for i in range(len(ind)) if ind[i]], [int(sum(ind) * N), p])
            Ystar = np.reshape([Y[i,:,:] for i in range(len(ind)) if ind[i]], [int(sum(ind) * N), q])

            inspect(Xstar.T @ Ystar)
            beta = (np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar)[1])) @ np.invert((Xstar.T @ Ystar))
            err = err + np.sqrt(sum(sum((Ystar - Xstar * beta) ** 2, 2)))

        err_best = err
        assig_best = assig
        for rep in range(1, repetitions):
            assig = np.zeros((ttrial, 1))
            while True:
                changes = np.cumsum(regularization + np.random.rand(1, K))
                changes = [1, np.floor(ttrial * changes / max(changes))]
                if ~any(changes == 0) and len(np.unique(changes)) == len(changes):
                    break
            err = 0

            for k in range(1, k + 1):
                assig[changes[k]:changes[k + 1]] = k
                ind = assig == k
                Xstar = np.reshape(X[ind, :, :], [sum(ind) * N, p])
                Ystar = np.reshape(Y[ind, :, :], [sum(ind) * N, q])
                beta = (np.transpose(Xstar) @ Xstar + 0.0001 * np.eye(np.shape(Xstar, 2))) @ np.invert((np.transpose(Xstar) @ Ystar))
                err = err + np.sqrt(sum(sum((Ystar - Xstar * beta) ** 2, 2)))

            if err < err_best:
                err_best = err
                assig_best = assig

        assig = assig_best

####### Fin Methode Sequential #######

####### Code de Fin #######

    else : #'fixedsequential'
        assig = np.ceil(K*[t/ttrial for t in range(1,ttrial)])


    Gamma = np.zeros((ttrial, K))
    for k  in range(K):
        #Gamma[assig==k,k] = 1
        Gamma[:,k] = [1 if a==k else None for a in assig]



    if swin > 1 :
        Gamma1 = Gamma
        Gamma = np.zeros((ttrial0-r,K))
        for k  in range(K):
            g = np.repmat(np.transpose(Gamma1[:,k]),[swin, 1])
            Gamma[:,k] = g[:]

        if r > 0 :
            Gamma = [[Gamma],
                     [np.repmat(Gamma[-1,:],[r, 1])]]


