# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann
'''

import numpy as np
import pandas as pd
import scipy.fft
import matplotlib.pylab as plt
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

'''
#################################
Batch Operators (input: List of arrays)
#################################
'''

def ecg_to_df(data, names):
    '''# Create pandas Dataframe columnweise

    Examples: ecg_to_df(ecg_leads, ecg_names)
    '''
    df = dict(zip(names, data))
    ecg_leads_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df.items()]))


'''
#################################
Single operators (input: array)
#################################
'''

'''
Normalize
'''

def ecg_norm(data):
    '''# Normalize
    Always first normalize, then denoise

    Examples: ecg_norm(ecg_leads[1])
    '''
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(data.reshape(-1, 1)).reshape(-1, )
    return scaled


def ecg_outlier(data, lower=0.001, upper=0.999):
    '''# Removes outliers by Z-Score
    Lower + Upper must equal 1!

    Examples: ecg_outlier(ecg_leads[1], lower=0.1, upper=0.9)
    ''' 
    index_up = np.nonzero(data > np.quantile(data, upper))
    index_low = np.nonzero(data < np.quantile(data, lower))

    data[index_up] = np.quantile(data, upper)
    data[index_low] = np.quantile(data, lower)

'''
Noise reduction
'''

def ecg_denoise_kalman(data, Q=1e-5, R=0.01):
    '''# Kalman Filter (best filter)
    Measurement variance R (lower=faster convergence)... should be between 0.01 – 1 
    Source: https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html

    Examples: ecg_denoise_kalman(ecg_leads[1])
    '''
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
    return yhat

def fft_lowpass(fft_centered, freq_idx, cutoff_freq=30, method='gauß'):
    '''# Lowpass-Filter in frequency domain
    Always run ecg_furier(..., center=True), then ecg_invfurier(..., center=True)!!
    "Hard" can create oscillations e.g. for 20-30hz

    Examples: fft_lowpass(data_ftt, freq, 40)
    '''
    if method == 'hard': 
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

def ecg_denoise_wavelet(data, level=2):
    '''# Denoise with Wavelet Decomposition
    Source: https: // www.section.io/engineering-education/wavelet-transform-analysis-of-1d-signals-using-python/

    Examples: ecg_denoise_wavelet(ecg_leads[1], 3)
    '''
    denoised = denoise_wavelet(data, wavelet='db6', mode='soft',
                               wavelet_levels=level, method='BayesShrink', rescale_sigma='True')
    return denoised

'''
More
'''

def ecg_furier(data, fs, lim=None, plot_out=False, center=True):
    '''# Furiertransformation

    Examples: ecg_furier(ecg_leads[1], fs, (1,300))
    '''
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

def ecg_invfurier(data, center=True):
    '''# Inv. Furiertransformation

    Examples: ecg_invfurier(data_fft)
    ''' 
    # Undo the shift to the center (if ecg_furier(..., center=True))
    if center == True:
        data = scipy.fft.ifftshift(data)
    
    # Inverse Transformation
    data_ifft = scipy.fft.ifft(data).real
    return data_ifft

def ecg_ceptrum(data, fs):
    '''# Ceptrum-Transformation

    Examples: ecg_ceptrum(data_fft, fs)
    '''
    data_ceptrum, freq = ecg_furier(data, fs, center=False)
    data_ceptrum = np.log(np.abs(data_ceptrum)**2)
    data_ceptrum, freq = ecg_furier(data_ceptrum, fs, center=False)
    data_ceptrum = np.abs(data_ceptrum)**2
    return data_ceptrum

def ecg_plot(data, start=0, end=None):
    '''# Time representation of data
    Start indicates, which index it starts (analogous for end).

    Examples: ecg_plot(ecg_leads[1])
    '''
    if end is None:
        end = data.shape[0]
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    axs[0].plot(np.arange(0, data.shape[0]), data)
    means = np.mean(data)
    axs[0].axhline(y=means)
    axs[0].set_xlim(start, end)

    axs[1].hist(data, bins=40)


def ecg_season_trend(data, plot_out=False):
    '''# Calculate trend & saisonality

    Examples: ecg_season_trend(ecg_leads[1])
    '''
    series = pd.Series(data)
    decompose_result = seasonal_decompose(series, period=fs, model="additive")

    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    if plot_out == True:
        decompose_result.plot()

    return trend, seasonal, residual