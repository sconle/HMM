# Storage management
import xarray as xr   # Manages .nc (netCDF) files in Python.
                      # The states' informations are stored in a .nc file for each subject.
from cluster_decoding import *

# Scientific computing
import numpy as np

data_dir = "D:/centrale/3A/info/HMM/data/"

subj=2
IC=1

ds = xr.open_dataset(data_dir + f"su{subj}IC{IC}_rawdata.nc")

print(ds['time'])


