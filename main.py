# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann
'''

# %%
from wettbewerb import load_references, save_predictions
from preprocess import *

from ecgdetectors import Detectors

# %%
# Import der EKG-Dateien
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Pipeline-Vorverarbeitung
ecg_leads_pre = [None]*len(ecg_leads)
ecg_leads_denoised = [None]*len(ecg_leads)
for i, dataset in enumerate(ecg_leads):
    ecg_leads_pre[i] = ecg_norm(ecg_leads[i])
    ecg_leads_denoised[i] = ecg_denoise_kalman(ecg_leads[i])
    print(i)

np.save("variables/ecg_leads_pre.npy", ecg_leads_pre)
np.save("variables/ecg_leads_denoised.npy", ecg_leads_denoised)

#%%
# Feature Extraction (Hamilton oder Pan-Tompkins)
detectors = Detectors(fs)
r_peaks = [None]*len(ecg_leads)
r_peaks_diff = [None]*len(ecg_leads)
for j, ecg_lead in enumerate(ecg_leads):
    r_peaks[j] = detectors.hamilton_detector(ecg_lead)
    r_peaks_diff[j] = np.diff(r_peaks[j])/fs*1000

np.save("variables/r_peaks.npy", r_peaks)
np.save("variables/r_peaks_diff.npy", r_peaks_diff)
