# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann
'''

# %%
from feature_extraction import *
from wettbewerb import load_references, save_predictions
from preprocess import *

# %%
# Import the data
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Pipeline-Preprocessing (first normalize, then denoise)
ecg_leads_norm = [None]*len(ecg_leads)
ecg_leads_detrend = [None]*len(ecg_leads)
ecg_leads_denoise = [None]*len(ecg_leads)
for i, dataset in enumerate(ecg_leads):
    trend, seasonal, residual = ecg_season_trend(ecg_leads[i], fs)
    ecg_leads_detrend[i] = ecg_leads[i] - trend

    ecg_leads_norm[i] = ecg_norm(ecg_leads_detrend[i])
    
    data, freq = ecg_furier(ecg_leads_norm[i], fs, center=True)
    data_den = ecg_denoise_spectrum(data, freq, 1, 30, "gauß")
    ecg_leads_denoise[i] = ecg_invfurier(data_den, center=True)

np.save("variables/ecg_leads_detrend.npy", ecg_leads_detrend)
np.save("variables/ecg_leads_norm.npy", ecg_leads_norm)
np.save("variables/ecg_leads_denoise.npy", ecg_leads_denoise)

# Feature Extraction (Hamilton or Pan-Tompkins)
peaks = [None]*len(ecg_leads)
peaks_diff = [None]*len(ecg_leads)
peaks_reshaped = [None]*len(ecg_leads)
peaks_std = [None]*len(ecg_leads)
for j, ecg_lead in enumerate(ecg_leads):
    peaks[j], peaks_diff[j] = ecg_detect(ecg_leads[j], fs, method="pan")
    peaks_diff[j] = ecg_norm(peaks_diff[j])
    peaks_reshaped[j], peaks_std[j] = ecg_poincare(peaks_diff[j], 3)

np.save("variables/r_peaks.npy", peaks)
np.save("variables/r_peaks_diff.npy", peaks_diff)
np.save("variables/peaks_reshaped.npy", peaks_reshaped)
np.save("variables/peaks_std.npy", peaks_std)

#%%

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(0,100):
    dimen = peaks_reshaped[i]
    std = peaks_std[i]
    if ecg_labels[i] == "N":
        color='green'
    if ecg_labels[i] == "A":
        color = 'red'
    if ecg_labels[i] == "~":
        color = 'gray'
    if ecg_labels[i] == "O":
        color = 'blue'
    ax.plot(dimen[:, 0], dimen[:, 1], std, color=color)
ax.view_init(0, 90)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# %%
