import numpy as np
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
        X = np.permute(X,[2, 1, 3, 4])
        X = np.reshape(X,[nwin, N*swin, p])
        Y = Y[to_use,:,:]
        Y = np.reshape(Y,[swin, nwin, N, q])
        Y = np.permute(Y,[2, 1, 3, 4])
        Y = np.reshape(Y,[nwin, N*swin, q])
        ttrial0 = ttrial; N0 = N
        ttrial = nwin; N = N*swin; T = nwin * np.ones((N,1))


####### Suite de Val ici ###########

####### Début Methode Hierarchical #######

    elif cluster_method == "hierarchical":
        beta = np.zeros((p, q), ttrial)

        for t in range(len(ttrial)):
            Xt = np.transpose(X[t, :, :], (1, 2, 0))
            Yt = np.transpose(Y[t, :, :], (1, 2, 0))
            beta[:, :, t] = (np.transpose(Xt) @ Xt) @ np.invert(np.transpose(Xt) @ Yt)

        if cluster_measure == "response":
            dist = np.zeros(ttrial*(ttrial-1)/2, 1)
            dist2 = np.zeros(ttrial, ttrial)
            Xstar = np.reshape(X,ttrial*N, p)  #Espace chelou sur matlab
            c = 1

            for t2 in range(0, ttrial):  #Est ce que la boucle doit se terminer à ttrial-1 ou ttrial ?
                d2 = Xstar * beta[:, :, t2]
                for t1 in range(t2, ttrial+1):  ## Idem que 2 lignes avant
                    d1 = Xstar * beta[:, :, t1]
                    dist[c] = np.sqrt(np.sum(np.sum((d1 - d2)**2)))
                    dist2[t1, t2] = dist[c]
                    dist2[t2, t1] = dist[c]
                    c += 1

        elif cluster_measure == "error":
            dist = np.zeros(ttrial * (ttrial - 1) / 2, 1)
            dist2 = np.zeros(ttrial, ttrial)
            c = 1

            for t2 in range(0, ttrial):  # Idem
                Xt2 = np.transpose(X[t2, :, :], (1, 2, 0))
                Yt2 = np.transpose(Y[t2, :, :], (1, 2, 0))
                for t1 in range(t2, ttrial+1):  # Idem
                    Xt1 = np.transpose(X[t1, :, :], (1, 2, 0))
                    Yt1 = np.transpose(Y[t1, :, :], (1, 2, 0))
                    error1 = np.sqrt(sum(sum((Xt1 * beta[:, :, t2] - Yt1)**2)))
                    error2 = np.sqrt(sum(sum((Xt2 * beta[:, :, t1] - Yt2) ** 2)))
                    dist[c] = error1 + error2
                    c += 1
                    dist2[t1, t2] = error1 + error2
                    dist[t2, t1] = error1 + error2

        elif cluster_measure == "beta":
            beta = np.transpose(beta, [2, 0, 1])
            beta = np.reshape(beta, [ttrial, p*q])  #Il manque  une virgule sur matlab
            dist = distance.pdist(beta)

        if distance.is_valid_dm(np.transpose(dist)):
            link = to_tree(linkage(np.transpose(dist), "ward")) ## A checker
        else:
            link = to_tree(linkage(np.transpose(dist)))

        assig = fcluster(link, criterion="maxclust", R=K)  #Est ce que c'est la bonne fct ?

####### Fin Methode Hierarchical #######

####### Début Methode Sequential #######

    elif cluster_method == "sequential":
        regularization = 1.0
        assig = np.zeros(ttrial, 1)
        err = 0
        changes = [i * np.floor(ttrial / K) for i in range(1, K)]
        Ystar = np.reshape(Y, [ttrial*N, q])

        for k in range(1, K+1):
            assig[changes[k]:changes[k+1]] = k
            ind = assig == k
            Xstar = np.reshape(X[ind, :, :], [sum(ind)*N, p])
            Ystar = np.reshape(Y[ind, :, :], [sum(ind)*N, q])
            beta = (Xstar.T @ Xstar + 0.0001 * np.eye(np.shape(Xstar, 2))) / (Xstar.T @ Ystar)
            err = err + np.sqrt(sum(sum((Ystar - Xstar * beta)**2, 2)))

        err_best = err
        assig_best = assig
        for rep in range(1, repetitions):
            assig = np.zeros(ttrial, 1)
            while True:
                changes = np.cumsum(regularization + np.random.rand(1, K))
                changes = [1, np.floor(ttrial * changes / max(changes))]
                if ~any(changes == 0) and len(np.unique(changes)) == len(changes):
                    break
            err = 0

            for k in range(1, k+1):
                assig[changes[k]:changes[k+1]] = k
                ind = assig == k
                Xstar = np.reshape(X[ind, :, :], [sum(ind)*N, p])
                Ystar = np.reshape(Y[ind, :, :], [sum(ind)*N, q])
                beta = (Xstar.T @ Xstar + 0.0001 * np.eye(np.shape(Xstar, 2))) / (Xstar.T @ Ystar)
                err = err + np.sqrt(sum(sum((Ystar - Xstar * beta) ** 2, 2)))

            if err < err_best:
                err_best = err
                assig_best = assig

        assig = assig_best

####### Fin Methode Sequential #######



print("Done")

