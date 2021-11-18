# Storage management
import xarray as xr   # Manages .nc (netCDF) files in Python.
                      # The states' informations are stored in a .nc file for each subject.

# Scientific computing
import numpy as np
from cluster_decoding import *
import os

#Le dossier "data" contenant les données doit se trouver dans le dossier mère

subj=2
IC=1

dirname = os.path.dirname(__file__)
filename = dirname + f"/data/su{subj}IC{IC}_rawdata.nc"
filename2 = dirname + f"/data/su{subj}IC{IC + 2}_rawdata.nc"

ds = xr.open_dataset(filename)
ds2 = xr.open_dataset(filename2)

X1 = ds['timecourse']
X2 = ds2['timecourse']

X = xr.concat([X1,X2], "new_dim")

print(X)


