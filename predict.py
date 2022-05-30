# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple

import preprocess
import scipy
import pickle
import hrv


###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='tree_model.sav',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier                                                 # Sampling-Frequenz 300 Hz
    hrz = hrv.HRV(fs)
    detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
    sdnn_array = np.array([])                                # Initialisierung der Feature-Arrays
    mnn_array = np.array([])
    rrskew_array = np.array([])
    rrkurt_array = np.array([])
    sdsd_array = np.array([])
    hr_array = np.array([])
    rmssd_array = np.array([])
    sdann_array = np.array([])
    pNN20_array = np.array([])
    pNN50_array = np.array([])
    NN20_array = np.array([])
    NN50_array = np.array([])
    for ecg_lead in ecg_leads:
            ecg_lead = preprocess.ecg_denoise_kalman(ecg_lead)
            r_peaks = detectors.pan_tompkins_detector(ecg_lead)     # Detektion der QRS-Komplexe
            #print(len(r_peaks))
            sdnn = np.std(np.diff(r_peaks)/fs*1000) 
            mnn = np.mean(np.diff(r_peaks)/fs*1000)            # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
            rrskew = scipy.stats.skew(np.diff(r_peaks)/fs*1000)
            rrkurt = scipy.stats.kurtosis(np.diff(r_peaks)/fs*1000)
            sdsd = hrz.SDSD(r_peaks)
            hr = hrz.HR(r_peaks)
            rmssd = hrz.RMSSD(r_peaks)
            if len(r_peaks)>1:
                pNN20 = hrz.pNN20(r_peaks)
                pNN50 = hrz.pNN50(r_peaks)
            else:
                pNN20 = np.nan
                pNN50 = np.nan
            NN20 = hrz.NN20(r_peaks)
            NN50 = hrz.NN50(r_peaks)
            #sdann = hrz.SDANN(r_peaks)
            sdnn_array = np.append(sdnn_array,sdnn)
            mnn_array = np.append(mnn_array, mnn)
            rrskew_array = np.append(rrskew_array, rrskew)
            rrkurt_array = np.append(rrkurt_array, rrkurt)
            sdsd_array = np.append(sdsd_array, sdsd)
            hr_array = np.append(hr_array, hr)
            rmssd_array = np.append(rmssd_array, rmssd)
            pNN20_array = np.append(pNN20_array, pNN20)
            pNN50_array = np.append(pNN50_array, pNN50)
            NN20_array = np.append(NN20_array, NN20)
            NN50_array = np.append(NN50_array, NN50)
            #sdann_array = np.append(sdann_array, sdann)

    inputArray = list(zip(sdnn_array, mnn_array, rrskew_array, rrkurt_array, sdsd_array, hr_array, rmssd_array, pNN20_array, pNN50_array, NN20_array, NN50_array))
    inputArray = np.array(inputArray)
    inputArray[np.where(np.isfinite(inputArray)==False)] = 0.0
    loaded_model = pickle.load(open(model_name, 'rb'))
    results =  loaded_model.predict(inputArray)
    result= []
    for k in range(len(results)):
        if results[k]== 0.0:
            result.append('N')
        else:
            result.append('A')
    predictions = list(zip(ecg_names, result))
    #print(predictions)

#------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!