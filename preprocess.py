# -*- coding: utf-8 -*-
"""
Preprocessing of the input data

@author: Julian Hüsselmann
"""
# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq
from skimage.restoration import denoise_wavelet
import scipy as scp
import matplotlib.pylab as plt

from wettbewerb import load_references, save_predictions

# %%
# Import der EKG-Dateien
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Spaltenweisen Datenframe erstellen
def ecg_to_df(data, names):  # z.B. ecg_to_df(ecg_leads, ecg_names)
    df = dict(zip(names, data))
    ecg_leads_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.items()]))

# Furieranalyse
def ecg_furier(data, lim=None, plot_out=False, center=True):  # z.B. ecg_furier(ecg_leads[1], (1,300))
    X = fft(data)

    N = len(X)
    n = np.arange(N)
    T = N/fs
    freq = n/T

    if center == True:  # Shift to the center
        X = scp.fft.fftshift(X)
        s = len(data)
        freq = np.arange(-s/2, s/2)*(fs/s)

    if plot_out == True:
        plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        if lim != None:
            plt.xlim(lim)
    return X

# Normieren 
def ecg_norm(data):  # z.B. ecg_norm(ecg_leads)
    for i in range(len(data)):
        data[i] = sc.fit_transform(data[i].reshape(-1, 1)).reshape(-1, )
    return data

# Denoise (Wavelet Decomp): https://www.section.io/engineering-education/wavelet-transform-analysis-of-1d-signals-using-python/
def ecg_denoise(data, level):  # z.B. ecg_denoise(ecg_leads, 2)
    for i in range(len(data)):
        data[i] = denoise_wavelet(data[i], wavelet='db6', mode='soft', wavelet_levels=level, method='BayesShrink', rescale_sigma='True')
    return data

# Zeitdarstellung
def ecg_plot(data, diag, start=0, end=None):  # z.B. ecg_plot(ecg_leads, 1, 0)
    if end is None:
        end = ecg_leads[diag].shape[0]
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    axs[0].plot(np.arange(0, ecg_leads[diag].shape[0]), ecg_leads[diag])
    means = np.mean(ecg_leads[diag])
    axs[0].axhline(y=means)
    axs[0].set_xlim(start, end)

    axs[1].hist(ecg_leads[diag], bins=40)

# Saison & Trend berechnen
def ecg_season_trend(data, plot_out=False):  # z.B. ecg_season_trend(ecg_leads[1])
    series = pd.Series(data)
    decompose_result = seasonal_decompose(series, period=fs, model="additive")

    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    if plot_out == True:
        decompose_result.plot()

    return trend, seasonal, residual

#%%
threshold = 3
outlier = []
for i in data:
    z = (i-mean)/std
    if z > threshold:
        outlier.append(i)
