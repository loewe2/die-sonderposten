# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann
'''

# %%
from ecg_noise_detector import noiseDetector
from wettbewerb import load_references, save_predictions
from preprocess import *

from ecgdetectors import Detectors

# %%
# Import the data
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Pipeline-Preprocessing (first normalize, then denoise)
ecg_leads_norm = [None]*len(ecg_leads)
ecg_leads_detrend = [None]*len(ecg_leads)
for i, dataset in enumerate(ecg_leads):
    ecg_leads_norm[i] = ecg_norm(ecg_leads[i])
    trend, seasonal, residual = ecg_season_trend(ecg_leads[i])
    ecg_leads_detrend[i] = ecg_leads_norm[i] - trend

np.save("variables/ecg_leads_norm.npy", ecg_leads_norm)
np.save("variables/ecg_leads_detrend.npy", ecg_leads_detrend)

#%%
# Feature Extraction (Hamilton or Pan-Tompkins)
detectors = Detectors(fs)
r_peaks = [None]*len(ecg_leads)
r_peaks_diff = [None]*len(ecg_leads)
for j, ecg_lead in enumerate(ecg_leads):
    r_peaks[j] = np.asarray(detectors.hamilton_detector(ecg_lead))
    r_peaks_diff[j] = np.diff(r_peaks[j])/fs*1000

np.save("variables/r_peaks.npy", r_peaks)
np.save("variables/r_peaks_diff.npy", r_peaks_diff)

#%%
a = ecg_leads[1]
start = 1000
end = 500
trend, seasonal, residual = ecg_season_trend(a, plot_out=True)
diff1 = a - trend
diff2 = a + residual
ecg_plot(a, start, end)
ecg_plot(diff1, start, end)
ecg_plot(diff2, start, end)

#%%
# Generate a noisy ECG
ecg = noiseDetector.get_example_ecg('noisy')

# Plot the ecg with green highlights on where clean signal is present
noiseDetector.plot_ecg(ecg)

# Classify the ecg
print(noiseDetector.is_noisy(ecg))
