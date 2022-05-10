# -*- coding: utf-8 -*-
"""
Preprocessing of the input data

@author: Julian Hüsselmann
"""
# %%
import numpy as np
import pandas as pd
import scipy.fft
import matplotlib.pylab as plt
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from wettbewerb import load_references, save_predictions

# %%
# Import der EKG-Dateien
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Batch-Verarbeitung:
# ffy, ffx = ecg_furier(ecg_leads[1], plot_out=False)
# filtered = fft_lowpass(ffy, ffx, 30, 'gauß')
# pure = ecg_invfurier(filtered)
# ecg_plot(pure, 2000, 3000)
# ecg_plot(ecg_leads[1], 2000, 3000)
for i, dataset in enumerate(ecg_leads):
    print(f"{i}: {fruit}")

#%%
"""
#################################
Batch Operators: List of arrays as input
#################################
"""
# Spaltenweisen Datenframe erstellen
def ecg_to_df(data, names):  # z.B. ecg_to_df(ecg_leads, ecg_names)
    df = dict(zip(names, data))
    ecg_leads_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.items()]))


"""
#################################
Single operators: Array as input
#################################
"""

"""
Normalize
"""
# Normalisieren (first normalize, then denoise)
def ecg_norm(data):  # z.B. ecg_norm(ecg_leads[1])
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(data.reshape(-1, 1)).reshape(-1, )
    return scaled

"""
Noise reduction
"""
# Kalman Filter (best filter): https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
# Measurement variance R(lower=faster convergence)... am besten zwischen 0.01 – 1
# z.B. ecg_denoise_kalman(ecg_leads[1])
def ecg_denoise_kalman(data, Q=1e-5, R=0.01):
    y = data
    x = np.arange(1, data.shape[0]+1)

    # intial parameters
    n_iter = sz = data.shape[0]

    # allocate space for arrays
    yhat = np.zeros(sz)      # a posteri estimate of x
    P = np.zeros(sz)         # a posteri error estimate
    yhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)    # a priori error estimate
    K = np.zeros(sz)         # gain

    # intial guesses
    yhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        yhatminus[k] = yhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/(Pminus[k]+R)
        yhat[k] = yhatminus[k]+K[k]*(y[k]-yhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return data

# Lowpass-Filter in frequency domain
# vorher: ecg_furier(..., center=True), danach ecg_invfurier(..., center=True)!!
def fft_lowpass(fft_centered, freq_idx, cutoff_freq=30, method='gauß'):  # z.B. fft_lowpass(data_ftt, freq, 40)
    if method == 'hard': # can create oscillations e.g. for 20-30hz
        #idx
        right_f = np.where(freq_idx > cutoff_freq)[0]
        left_f = np.where(freq_idx < -cutoff_freq)[0]

        # Filtering
        fft_centered[right_f] = 0
        fft_centered[left_f] = 0
    elif method == 'gauß':
        gauß_k = np.exp(-np.power(freq_idx - 0, 2.) / (2 * np.power(cutoff_freq, 2.)))
        fft_centered = fft_centered*gauß_k
    
    fft_filtered = fft_centered
    return fft_filtered

# Denoise with Wavelet Decomposition: https: // www.section.io/engineering-education/wavelet-transform-analysis-of-1d-signals-using-python/
# z.B. ecg_denoise_wavelet(ecg_leads[1], 3)
def ecg_denoise_wavelet(data, level=2):
    denoised = denoise_wavelet(data, wavelet='db6', mode='soft',
                               wavelet_levels=level, method='BayesShrink', rescale_sigma='True')
    return denoised

"""
More
"""
# Furiertransformation
# z.B. ecg_furier(ecg_leads[1], (1,300))
def ecg_furier(data, lim=None, plot_out=False, center=True):
    # Transformation
    data_ftt = scipy.fft.fft(data)

    # Initial Values
    N = len(data_ftt)
    n = np.arange(N)
    T = N/fs
    freq = n/T

    # Shift to the center
    if center == True:
        data_ftt = scipy.fft.fftshift(data_ftt)
        s = len(data)
        freq = np.arange(-s/2, s/2)*(fs/s)

    # Plot
    if plot_out == True:
        plt.stem(freq, np.abs(data_ftt), 'b', markerfmt=" ", basefmt="-b")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        if lim != None:
            plt.xlim(lim)
    return data_ftt, freq

# Inv. Furiertransformation
def ecg_invfurier(data, center=True):  # z.B. ecg_invfurier(data_fft)
    # Undo the shift to the center (if ecg_furier(..., center=True))
    if center == True:
        data = scipy.fft.ifftshift(data)
    
    # Inverse Transformation
    data_ifft = scipy.fft.ifft(data).real
    return data_ifft

# Time representation of data
def ecg_plot(data, start=0, end=None):  # z.B. ecg_plot(ecg_leads[1])
    if end is None:
        end = data.shape[0]
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    axs[0].plot(np.arange(0, data.shape[0]), data)
    means = np.mean(data)
    axs[0].axhline(y=means)
    axs[0].set_xlim(start, end)

    axs[1].hist(data, bins=40)

# Calculate trend & saisonality
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

#%%
# Data Generator

# start=-0.5
# end=1.5
# steps = 100

# sigma = 0.3
# level=5

# x = np.arange(start, end, 1/steps)
# y1 = np.exp(2*x) + sigma*np.random.randn(x.shape[0])
# y2 = np.exp(0.1*x)+2 + sigma*np.random.randn(x.shape[0])

# plt.plot(x, y)
# plt.plot(x, yhat)
# plt.savefig('/home/tim/Schreibtisch/filename.png', dpi=600)

# %%
aa = 9

plt.hist(ecg_leads[a])

# %%
