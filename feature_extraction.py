# -*- coding: utf-8 -*-
'''
Different methods to extract the features

@author: Julian Hüsselmann
'''

import numpy as np
from ecgdetectors import Detectors

# QRS-Detector
def ecg_detect(data, fs, method="pan"):
    '''# Detect QRS-complexes

    Examples: ecg_plot(ecg_leads[1])
    '''
    detectors = Detectors(fs)
    if method == "hamilton":
        peaks = detectors.hamilton_detector(data)
    elif method == "christov":
        peaks = detectors.christov_detector(data)
    elif method == "engzee":
        peaks = detectors.engzee_detector(data)
    elif method == "pan":
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
    diff = np.diff(peaks)/fs*1000

    return peaks, diff

def ecg_poincare(peaks_diff, dim=2):
    '''# Assign a dimension to each difference and repeat it

    Examples: ecg_poincare(r_peaks_diff[1], 2)
    '''

    shape = peaks_diff.shape
    full, rest = divmod(shape[0], dim)
    reshaped = peaks_diff[0:full*dim].reshape((full, dim))

    std = np.std(peaks_diff)
    return reshaped, std
