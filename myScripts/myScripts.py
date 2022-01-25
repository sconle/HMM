# Storage management
import matplotlib.pyplot as plt
import xarray as xr   # Manages .nc (netCDF) files in Python.
                      # The states' informations are stored in a .nc file for each subject.

# Scientific computing
import numpy as np
import os
import sys
sys.path.append(r"D:\centrale\3A\info\HMM\myHmmPackage")
from myHmmPackage.cluster_decoding import *
from myHmmPackage.cluster_decoder import *
from sklearn.utils.estimator_checks import check_estimator
from myHmmPackage.tde_hmm import *


#Le dossier "data" contenant les données doit se trouver dans le dossier mère

subj=2
IC=1

dirname = os.path.dirname(__file__ + "/../../")
filename = dirname + f"/data/su{subj}IC{IC}_rawdata.nc"
filename2 = dirname + f"/data/su{subj}IC{IC + 2}_rawdata.nc"
ds = xr.open_dataset(filename)
ds2 = xr.open_dataset(filename2)
X1 = np.transpose(ds['timecourse'].values)[:,:,np.newaxis]
X2 = np.transpose(ds2['timecourse'].values)[:,:,np.newaxis]

X = np.concatenate((X1, X2), axis=2)

# trialinfo = ds['trialinfo']
# y = ((trialinfo/10000).astype(int) == 1)
# [ttrial, N, p] = np.shape(X)
# Y = np.ones((ttrial, N, 1))
# for i in range(ttrial):
#     Y[i]=y
#
# X = np.reshape(X, (ttrial*N, p))

TEST = "cluster_decoder"

if TEST == "cluster_decoding":
    #Le dossier "data" contenant les données doit se trouver dans le dossier mère

    subj=2
    IC=1

    dirname = os.path.dirname(__file__ + "/../../")
    filename = dirname + f"/data/su{subj}IC{IC}_rawdata.nc"
    filename2 = dirname + f"/data/su{subj}IC{IC + 2}_rawdata.nc"

    ds = xr.open_dataset(filename)
    ds2 = xr.open_dataset(filename2)

    X1 = np.transpose(ds['timecourse'].values)[:,:,np.newaxis]
    X2 = np.transpose(ds2['timecourse'].values)[:,:,np.newaxis]


    X = np.concatenate((X1, X2), axis=2)

    trialinfo = ds['trialinfo']
    y = ((trialinfo/10000).astype(int) == 1)
    [ttrial, N, p] = np.shape(X)
    Y = np.ones((ttrial, N, 1))
    for i in range(ttrial):
        Y[i]=y



    X = np.reshape(X, (ttrial*N, p))
    Y = np.reshape(Y, (ttrial*N, 1))
    T = (np.ones((N, 1)))*ttrial
    K = 3
    cluster_method='sequential'
    Gamma = cluster_decoding(X, Y, T, K, cluster_method)

    print(Gamma)
    plt.imshow(Gamma.T, aspect='auto')
    plt.show()

elif TEST == "cluster_decoder":
    decoder = ClusterDecoder(method='regression',
                             init_scheme=np.array([0, 0, 0, 1]),
                             transition_scheme=[[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]])

    subj = 2
    IC = 1

    dirname = os.path.dirname(__file__ + "/../../")
    filename = dirname + f"/data/su{subj}IC{IC}_rawdata.nc"
    filename2 = dirname + f"/data/su{subj}IC{IC + 2}_rawdata.nc"

    ds = xr.open_dataset(filename)
    ds2 = xr.open_dataset(filename2)

    X1 = ds['timecourse'].values[:, :, np.newaxis]
    X2 = ds2['timecourse'].values[:, :, np.newaxis]

    X = np.concatenate((X1, X2), axis=2)

    trialinfo = ds['trialinfo']
    y_ = ((trialinfo / 10000).astype(int) == 1)
    [n_samples, n_time_points, n_regions] = np.shape(X)
    y = np.ones((n_samples, n_time_points, 1))
    for i in range(n_time_points):
        y[:,i] = y_

    decoder.fit(X,y)
    plt.imshow(decoder.gamma_.T, aspect='auto')
    plt.show()

elif TEST == "tde_hmm":
    # On récupère un signal
    X = X[:, :, 0]

    # On concatène pour obtenir qu'un seul array
    X = np.reshape(X, X.shape[0] * X.shape[1])
    X = np.reshape(X, -1)

    # On découpe le signal en plusieurs sous signaux décalés d'un cran
    n_fenetre = 10
    n = len(X)
    downsamp_rate = 3
    signal_crante = np.ones((n-n_fenetre, n_fenetre))

    for i in range(n-n_fenetre):
        signal_crante[i] = X[i:i+n_fenetre]

    # signal_crante = np.reshape(signal_crante, (-1, 1))

    n_states = 3
    n_iter = 1000000
    covariance_type = 'full'
    tol = 0.01
    hmm = TDE_HMM(n_components=n_states, n_iter=n_iter, covariance_type=covariance_type, tol=tol)
    hmm.fit(signal_crante)
    print(hmm.predict_proba(signal_crante))

    print('FIN')

