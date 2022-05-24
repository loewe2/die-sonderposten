# -*- coding: utf-8 -*-
'''
Different methods to extract the features

@author: Julian Hüsselmann
'''

import numpy as np
from ecgdetectors import Detectors
from preprocess import *

'''
Detection
'''

def ecg_detect(data, fs, method="pan"):
    '''# Detect QRS-complexes

    Examples: ecg_plot(ecg_leads[1])
    '''
    detectors = Detectors(fs)
    if method == "hamilton": # TOP
        peaks = detectors.hamilton_detector(data)
    elif method == "christov":
        peaks = detectors.christov_detector(data)
    elif method == "engzee": #Detects the exact point on R-waves the best
        peaks = detectors.engzee_detector(data)
    elif method == "pan": #TOP
        peaks = detectors.pan_tompkins_detector(data)
    elif method == "wavelet":
        peaks = detectors.swt_detector(data)
    elif method == "average":
        peaks = detectors.two_average_detector(data)
    elif method == "matched":
        peaks = detectors.matched_filter_detector(data)
    elif method == "wqrs":
        peaks = detectors.wqrs_detector(data)
    
    peaks = np.asarray(peaks)
    diff = np.diff(peaks)

    return peaks, diff


'''
Alteration
'''

def ecg_poincare(peaks_diff, dim=2):
    '''# Assign a dimension to each difference and repeat it
    Columnwise x, y, z, ... dimensions
    Examples: ecg_poincare(r_peaks_diff[1], 2)
    '''

    shape = peaks_diff.shape
    full, rest = divmod(shape[0], dim)
    reshaped = peaks_diff[0:full*dim].reshape((full, dim))

    std = np.std(peaks_diff)
    return reshaped, std

def ecg_snippets(data, peaks, peaks_diff, plot_out=False):
    '''# Segmentation: Creates time based snippets (cuts the whole signal in single periods)

    Examples: ecg_snippets(ecg_leads[1], peaks[1], peaks_diff[1], plot_out=False)
    '''
    # Define borders left and right of each detected peak (half of every difference between the peaks)
    left_side, right_side = np.asarray(ecg_empty(peaks, 2))
    for m, diff in enumerate(peaks):
        if m == 0:
            left_side[m] = peaks[m]-0.5*peaks_diff[m]
        else:
            left_side[m] = peaks[m]-0.5*peaks_diff[m-1]
        if m == (peaks.size-1):
            right_side[m] = peaks[m]+0.5*peaks_diff[m-1]
        else:
            right_side[m] = peaks[m]+0.5*peaks_diff[m]

    # Replace (unrealistic) indices that are negative or bigger than the original "peaks"-data
    limit_l = np.where(left_side < 0)
    left_side[limit_l] = 0
    limit_r = np.where(right_side > np.max(peaks))
    right_side[limit_r] = np.max(peaks)

    # Convert to other datatype
    left_side = left_side.astype(np.int64)
    right_side = right_side.astype(np.int64)

    # Get peak snippets by the right and left border indices
    period_container = ecg_empty(peaks, 1)
    time_container = ecg_empty(peaks, 1)
    for i, content in enumerate(peaks):
        idx1 = left_side[i]
        idx2 = right_side[i]
        period_container[i] = data[idx1:idx2]
        time_container[i] = left_side[i], right_side[i]

    # Plot
    if plot_out == True:
        for j, data in enumerate(period_container):
            x = np.arange(period_container[j].size)
            y = period_container[j]
            plt.plot(x, y)

    return period_container, time_container

def ecg_diff_mean(data):
    '''# Calculates the centroid (mean), row-wise
    Examples: reject_outliers(peaks_reshaped[0])
    '''
    mean = np.mean(data, axis=0)
    return mean

'''
Plot
'''

def plot_diff(peaks_reshaped, peaks_std, start, end, ecg_labels, azim, elev):
    '''# Plot the feature space

    Examples: plot_diff(peaks_reshaped, peaks_std, 0, 100, ecg_labels, 0, 90)
    '''
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(start, end):
        if ecg_labels[i] == "N":
            color = 'green'
        if ecg_labels[i] == "A":
            color = 'red'
        if ecg_labels[i] == "~":
            color = 'gray'
        if ecg_labels[i] == "O":
            color = 'blue'
        x = peaks_reshaped[i][:, 0]
        y = peaks_reshaped[i][:, 1]
        z = peaks_std[i]
        ax.scatter(x, y, z, color=color)
    ax.view_init(azim, elev)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return None