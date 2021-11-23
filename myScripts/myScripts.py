# Storage management
import xarray as xr   # Manages .nc (netCDF) files in Python.
                      # The states' informations are stored in a .nc file for each subject.

# Scientific computing
from myHmmPackage.cluster_decoding import *
import os

#Le dossier "data" contenant les données doit se trouver dans le dossier mère

subj=2
IC=1

dirname = os.path.dirname(__file__)
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

print(Y.shape)

X = np.reshape(X, (ttrial*N, p))
Y = np.reshape(Y, (ttrial*N, 1))
T = (np.ones((N, 1)))*ttrial
K = 6
cluster_method='regression'
Gamma = cluster_decoding(X, Y, T, K, cluster_method)
