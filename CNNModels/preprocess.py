# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann, ...
'''

#%%
import numpy as np
import pandas as pd
import scipy.fft
import matplotlib.pylab as plt
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import interp1d
from scipy import signal

'''
Normalize & Outlier
'''

def ecg_norm(data):
    '''# Normalize
    Always first normalize, then denoise

    Examples: ecg_norm(ecg_leads[1])
    '''
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(data.reshape(-1, 1)).reshape(-1, )
    return scaled

def pandas_normalize(data):
    '''# Normalize data

    Examples: pandas_normalize(pd)
    '''
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)
    normalized = pd.DataFrame(scaled, columns=data.columns)
    return normalized

def ecg_outlier(data, lower=0.001, upper=0.999):
    '''# Removes outliers by Z-Score
    Lower + Upper must equal 1!

    Examples: ecg_outlier(ecg_leads[1], lower=0.1, upper=0.9)
    ''' 
    index_up = np.nonzero(data > np.quantile(data, upper))
    index_low = np.nonzero(data < np.quantile(data, lower))

    data[index_up] = np.quantile(data, upper)
    data[index_low] = np.quantile(data, lower)

def reject_outliers(data, m=2):
    '''# Remove outliers (only for vectors)
    Source: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    Examples: reject_outliers(peaks_reshaped[0][1])
    '''
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def ecg_season_trend(data, fs, plot_out=False):
    '''# Calculate trend & saisonality
    Initial outliers can be removed by substracting
    the trend from the original data (alternatively: add the risidual to the original data)

    Examples: ecg_season_trend(ecg_leads[1], fs)
    '''
    series = pd.Series(data)
    decompose_result = seasonal_decompose(series, period=fs, model="additive")

    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    if plot_out == True:
        decompose_result.plot()

    return trend, seasonal, residual

'''
Noise reduction
'''

def ecg_denoise_kalman(data, Q=1e-5, R=0.01):
    '''# Kalman Filter
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

def ecg_denoise_spectrum(fft_centered, freq_idx, lower_freq=0, upper_freq=20, method='gauß'):
    '''# Bandpass-Filter in frequency domain
    Always run ecg_furier(..., center=True) before and ecg_invfurier(..., center=True) after!!
    "Hard" can create oscillations around 20Hz

    Examples: ecg_denoise_spectrum(data_ftt, freq, 20, 50)
    '''
    #Create Index-Vector (0: block, 1: pass frequencies)
    right_p = np.where(freq_idx > upper_freq)[0]
    left_p = np.where((0 > freq_idx) & (freq_idx < lower_freq))[0]
    right_n = np.where((-lower_freq < freq_idx) & (freq_idx < 0))[0]
    left_n = np.where(freq_idx < -upper_freq)[0]

    idx_shape = np.ones(freq_idx.shape)
    idx_shape[right_p] = 0
    idx_shape[left_p] = 0
    idx_shape[right_n] = 0
    idx_shape[left_n] = 0

    # Filtering
    if method == 'hard':
        fft_centered = fft_centered*idx_shape
    elif method == 'gauß':
        cutoff_freq = np.abs(upper_freq - lower_freq)
        mean = lower_freq+(upper_freq-lower_freq)/2

        gauß_l = np.exp(-np.power(freq_idx + mean, 2.) /
                        (2 * np.power(cutoff_freq, 2.)))
        gauß_r = np.exp(-np.power(freq_idx - mean, 2.) /
                        (2 * np.power(cutoff_freq, 2.)))
        gauß = gauß_l+gauß_r
        fft_centered = fft_centered*gauß
    
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
Spectrum
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
    Investigate periodic structures in frequency spectra

    Examples: ecg_ceptrum(data_fft, fs)
    '''
    data_ceptrum, freq = ecg_furier(data, fs, center=False)
    data_ceptrum = np.log(np.abs(data_ceptrum)**2)
    data_ceptrum, freq = ecg_furier(data_ceptrum, fs, center=False)
    data_ceptrum = np.abs(data_ceptrum)**2
    return data_ceptrum

'''
More
'''

def resample(data, res_factor, method='cubic'): 
    '''# Undersamples/oversamples the data
    
    res_factor of 0.5 e.g. downsamples from 300 Hz to 150 Hz

    Source: https://stackoverflow.com/questions/29085268/resample-a-numpy-array
    
    Methods:
    - linear, slinear, cubic, quadratic
    - furier (can result in problems/oscillations)
    - nearest, nearest-up, previous, next
    - zero

    Examples: resample(ecg_leads[2], 0.5, 'linear')
    '''
    if method == 'furier':
        resampled = signal.resample(data, int(len(data)*res_factor))
    else:
        xp = np.arange(0, len(data), 1/res_factor)
        nearest = interp1d(np.arange(len(data)), data, kind='cubic')
        resampled = nearest(xp)
    return resampled

def ecg_empty(size, number):
    '''# Creates an empty array based on size
    Number must equal the output variables

    Examples: peaks, std, st1, st2 = ecg_empty(ecg_leads, 4)
    '''
    container = [None]*number
    if number > 1:
        for i in range(0, number):
            container[i] = [None]*len(size)
    else:
        container = [None]*len(size)
    return container

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
    axs[0].axhline(y=means, color="gray", linestyle="--")
    axs[0].set_xlim(start, end)

    axs[1].hist(data, bins=40)
    return None