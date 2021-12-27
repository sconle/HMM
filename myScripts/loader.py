# import h5py
import numpy as np
import mne
from wavelet_transform import *

def load_oneIC(mat_file, cells_refs, subj=2, IC=1, comp=False):

    data = {}
    
    try:
        print("Loading the raw timecourse")
        data[f'raw_timecourse_256Hz'] = _load_raw_timecourse_256Hz(mat_file, cells_refs, IC-1, subj-1)
        data['trialinfo'] = _load_trialinfo(mat_file, cells_refs, IC-1, subj-1)
        
        n_trials = data[f'raw_timecourse_256Hz'].shape[0]
    
        data['freq_axis'] = _load_freq(mat_file, cells_refs, IC-1, subj-1)
        data['time_axis'] = _load_time_256Hz(mat_file, cells_refs, IC-1, subj-1)

        info = mne.create_info(ch_names=['signal'], sfreq=256, ch_types=['eeg'])

        if comp==True:
            print(f"Computing and loading the time-frequency wavelet transformation for 3 trials")
            for trial in range(1, 4):
                data[f'tfr_256Hz trial{trial}'] = wavelet_transform2(data, info, trial)

    except:
        print(f'The independent component IC{IC} of the subject {subj} is not in the .mat file.')
        raise

    return data, n_trials
    

def _load_raw_timecourse_256Hz(mat_file, cells_refs, IC, subj):
    cell = mat_file[cells_refs[IC, subj]]
    raw_timecourse_256Hz = cell['raw_timecourse_256Hz'][:]
    raw_timecourse_256Hz = np.transpose(raw_timecourse_256Hz)
    return raw_timecourse_256Hz

def _load_trialinfo(mat_file, cells_refs, IC, subj):
    cell = mat_file[cells_refs[IC, subj]]
    trialinfo = cell['trialinfo'][:]
    trialinfo = np.transpose(trialinfo)
    return trialinfo

def _load_time_256Hz(mat_file, cells_refs, IC, subj):
    cell = mat_file[cells_refs[IC, subj]]
    time_256Hz = cell['time_256Hz'][:]
    time_256Hz = np.transpose(time_256Hz)[0]
    return time_256Hz

def _load_freq(mat_file, cells_refs, IC, subj):
    cell = mat_file[cells_refs[IC, subj]]
    freq = cell['freq'][:]
    freq = np.transpose(freq)[0]
    return freq

def _load_ALL_FREQ_power_20Hz(mat_file, cells_refs, IC, subj):
    cell = mat_file[cells_refs[IC, subj]]
    all_freq_power_20Hz = cell['ALL_FREQ_power_20Hz'][:]
#     all_freq_power_20Hz = np.transpose(all_freq_power_20Hz)
    return all_freq_power_20Hz
            


