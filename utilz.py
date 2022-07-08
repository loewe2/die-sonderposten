from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import scipy
import pandas as pd

def predictions_score(y_true_path, y_pred_path):

    y_true_all = read_csv(y_true_path, delimiter=',', header=None)
    y_pred_all = read_csv(y_pred_path, delimiter=',', header=None)
    y_true_all = y_true_all[1].values
    y_pred_all = y_pred_all[1].values
    y_true = []
    y_pred = []
    for i in range(len(y_true_all)):
        if not(y_true_all[i] == '~' or y_true_all[i] == 'O'):
            y_true.append(y_true_all[i])
            y_pred.append(y_pred_all[i])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    enc = OneHotEncoder()
    enc.fit(y_true.reshape(-1,1))
    y_true = enc.transform(y_true.reshape(-1,1)).toarray()
    y_pred = enc.transform(y_pred.reshape(-1,1)).toarray()

    cm = confusion_matrix(y_true[:,0], y_pred[:,0])
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    y_pred_list = y_pred[:,0].tolist()
    print(y_pred_list.count(1))

    return (f1, acc, cm)

def calcPeakStatistic(indexes_1,signal, name):
    return calcStatisticalFeatures(signal[indexes_1], name)

def calcPeakRelationStatistic(indexes_1, indexes_2, signal, inverse, name):
    return calcStatisticalFeatures(amplitudeOfPeak(indexes_1, indexes_2, signal, inverse), name)

def amplitudeOfPeak(indexes_1, indexes_2, signal, inverse=False):
    #indexes_1 is the leading signal event
    amplitude = []
    for i in range(len(indexes_1)):
        current_index = indexes_1[i]
        if(indexes_1[-1] == indexes_1[i]):
            next_index = len(signal) - 1
        else:
            next_index = indexes_1[i+1]
        pos_indexes = np.where((indexes_2 >= current_index) & (indexes_2 <= next_index))
        pos_indexes = pos_indexes[0]
        if not(len(pos_indexes) == 0):
            relation = signal[indexes_2[pos_indexes[0]]]/(signal[current_index]+1e12)
            if(inverse):
                relation = 1/relation
            amplitude.append(relation)
    return amplitude

def calcDiff(indexes_1, indexes_2, signal_len):
    #indexes_1 is the leading signal event
    diff_list = []
    for i in range(len(indexes_1)):
        current_index = indexes_1[i]
        if(indexes_1[-1] == indexes_1[i]):
            next_index = signal_len - 1
        else:
            next_index = indexes_1[i+1]
        pos_indexes = np.where((indexes_2 >= current_index) & (indexes_2 <= next_index))
        pos_indexes = pos_indexes[0]
        if not(len(pos_indexes) == 0):
            diff = indexes_2[pos_indexes[0]]-current_index
            diff_list.append(diff)
    return diff_list

def calcFeaturesFromPeaks(indexes_1, indexes_2, signal_len,name):
    return calcStatisticalFeatures(calcDiff(indexes_1, indexes_2, signal_len),name)   

def calcStatisticalFeatures(data, name):
    name = str(name)
    if(len(data) > 0):
        mean = np.mean(data)
        min  = np.min(data)
        max  = np.max(data)
        std  = np.std(data)
        skewness = scipy.stats.skew(data)
        kurtosis = scipy.stats.kurtosis(data)
    else:
        mean = np.nan
        min = np.nan
        max = np.nan
        std = np.nan
        skewness = np.nan
        kurtosis = np.nan
    return {name+'Mean':mean, name+'Min':min, name+'Max':max, name+'Std':std, name+'Skew':skewness, name+'Kurt':kurtosis}

def ownFeatures(signals):
    index_Q_Peaks = np.where(signals['ECG_Q_Peaks'].values == 1)[0]
    index_T_Peaks = np.where(signals['ECG_T_Peaks'].values == 1)[0]
    index_S_Peaks = np.where(signals['ECG_S_Peaks'].values == 1)[0]
    index_R_Peaks = np.where(signals['ECG_R_Peaks'].values == 1)[0]
    index_P_Peaks = np.where(signals['ECG_P_Peaks'].values == 1)[0]
    index_P_Onsets = np.where(signals['ECG_P_Onsets'].values == 1)[0]
    index_T_Onsets = np.where(signals['ECG_T_Onsets'].values == 1)[0]

    signal_len = len(signals['ECG_Raw'])

    PQ = calcFeaturesFromPeaks(index_P_Peaks, index_Q_Peaks, signal_len, 'PQ')
    PR = calcFeaturesFromPeaks(index_P_Peaks, index_R_Peaks, signal_len, 'PR')
    PS = calcFeaturesFromPeaks(index_P_Peaks, index_S_Peaks, signal_len, 'PS')
    PT = calcFeaturesFromPeaks(index_P_Peaks, index_T_Peaks, signal_len, 'PT')
    QR = calcFeaturesFromPeaks(index_Q_Peaks, index_R_Peaks, signal_len, 'QR')
    QS = calcFeaturesFromPeaks(index_Q_Peaks, index_S_Peaks, signal_len, 'QS')
    QT = calcFeaturesFromPeaks(index_Q_Peaks, index_T_Peaks, signal_len, 'QT')
    RS = calcFeaturesFromPeaks(index_R_Peaks, index_S_Peaks, signal_len, 'RS')
    RT = calcFeaturesFromPeaks(index_R_Peaks, index_T_Peaks, signal_len, 'RT')
    ST = calcFeaturesFromPeaks(index_S_Peaks, index_T_Peaks, signal_len, 'ST')

    RdP = calcPeakRelationStatistic(index_P_Peaks, index_R_Peaks, signals['ECG_Clean'], True, 'RdP')
    RdQ = calcPeakRelationStatistic(index_Q_Peaks, index_R_Peaks, signals['ECG_Clean'], True, 'RdQ')
    RdT = calcPeakRelationStatistic(index_R_Peaks, index_T_Peaks, signals['ECG_Clean'], False, 'RdT')
    RdS = calcPeakRelationStatistic(index_R_Peaks, index_S_Peaks, signals['ECG_Clean'], False, 'RdS')

    PStat = calcPeakStatistic(index_P_Peaks,signals['ECG_Clean'], 'PStat')
    RStat = calcPeakStatistic(index_R_Peaks,signals['ECG_Clean'], 'RStat')
    SStat = calcPeakStatistic(index_S_Peaks,signals['ECG_Clean'], 'SStat')
    QStat = calcPeakStatistic(index_Q_Peaks,signals['ECG_Clean'], 'QStat')
    TStat = calcPeakStatistic(index_T_Peaks,signals['ECG_Clean'], 'TStat')

    return pd.DataFrame({**PQ, **PR, **PS, **PT, **QR, **QS, **QT, **RS, **RT, **ST, **RdP, **RdQ, **RdT, **RdS, **PStat, **QStat, **RStat, **SStat, **TStat}, index=[0])
    