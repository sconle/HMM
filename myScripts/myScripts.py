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

TEST = "tde_hmm"

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
    decoder = ClusterDecoder(max_iter=10, method='sequential')

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
    print(1 in decoder.predict(X))
    plt.imshow(decoder.gamma_.T, aspect='auto')
    plt.show()

elif TEST == "tde_hmm":

    hmm = TDE_HMM()
    hmm.fit(X)
    print(hmm.predict_proba(X))

    print('FIN')

