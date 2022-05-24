# -*- coding: utf-8 -*-
'''
Main script

@author: Julian Hüsselmann
'''

# %%
from feature_extraction import *
from wettbewerb import load_references, save_predictions
from preprocess import *
from tensorflow import keras

# %%
# Import the data
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Pipeline-Preprocessing (first normalize, then denoise)
ecg_leads_norm, ecg_leads_detrend, ecg_leads_denoise = ecg_empty(ecg_leads, 3)
for i, dataset in enumerate(ecg_leads):
    trend, seasonal, residual = ecg_season_trend(ecg_leads[i], fs)
    ecg_leads_detrend[i] = np.asarray(ecg_leads[i] - trend)

    ecg_leads_norm[i] = ecg_norm(ecg_leads_detrend[i])

    data, freq = ecg_furier(ecg_leads_norm[i], fs, center=True)
    data_den = ecg_denoise_spectrum(data, freq, 1, 25, "gauß")
    ecg_leads_denoise[i] = ecg_invfurier(data_den, center=True)
    print("Pipeline:"+i)

# Feature Extraction (Hamilton or Pan-Tompkins)
peaks, peaks_diff, peaks_diff_norm, peaks_diff_mean, peaks_reshaped, ecg_std, peaks_std = ecg_empty(
    ecg_leads, 7)
for j, ecg_lead in enumerate(ecg_leads):
    peaks[j], peaks_diff[j] = ecg_detect(ecg_leads[j], fs, method="pan")
    # Don't apply calculations to signals, where no QRS-complexes have been found
    if peaks_diff[j].size != 0:
        peaks_diff_norm[j] = ecg_norm(peaks_diff[j])
        peaks_reshaped[j], peaks_std[j] = ecg_poincare(peaks_diff_norm[j], 3)
        ecg_std[j] = np.std(ecg_leads[j])
        peaks_diff_mean[j] = np.std(ecg_leads[j])
    else:
        peaks_reshaped[j], peaks_std[j], ecg_std[j] = None, None, None
    print("Pipeline:"+j)

# Variables
###
np.save("variables/ecg_leads_detrend.npy", ecg_leads_detrend)
np.save("variables/ecg_leads_norm.npy", ecg_leads_norm)
np.save("variables/ecg_leads_denoise.npy", ecg_leads_denoise)
###
np.save("variables/peaks.npy", peaks)
np.save("variables/peaks_diff.npy", peaks_diff)
np.save("variables/peaks_diff_norm.npy", peaks_diff_norm)
np.save("variables/peaks_diff_mean.npy", peaks_diff_mean)
np.save("variables/peaks_reshaped.npy", peaks_reshaped)
np.save("variables/peaks_std.npy", peaks_std)
np.save("variables/ecg_std.npy", ecg_std)
#%%
peaks_reshaped = np.load("variables/peaks_reshaped.npy", allow_pickle=True)
peaks_diff = np.load("variables/peaks_diff.npy", allow_pickle=True)

#%%
def ecg_diff_mean(data):
    '''# Calculates the centroid (mean), row-wise
    Examples: reject_outliers(peaks_reshaped[0])
    '''
    mean = np.mean(data, axis=0)
    return mean

fig = plt.figure()
for i in range(1,600):
    aa = peaks_diff[i]
    peaks_diff_norm = ecg_norm(aa)
    peaks_reshaped, peaks_std = ecg_poincare(peaks_diff_norm, 2)
    peaks_reshaped_x = peaks_reshaped[:, 0]
    peaks_reshaped_y = peaks_reshaped[:, 1]
    x = ecg_diff_mean(peaks_reshaped_x)
    y = ecg_diff_mean(peaks_reshaped_y)
    if ecg_labels[i] == "N":
        color = 'green'
    if ecg_labels[i] == "A":
        color = 'red'
    if ecg_labels[i] == "~":
        color = 'white'
    if ecg_labels[i] == "O":
        color = 'white'
    plt.scatter(x, y, color=color)


#%%
# Classificator
