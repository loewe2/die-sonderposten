import csv
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from wettbewerb import load_references
import tensorflow as tf
import tensorflow.keras as keras
import icentiaDataProcessor
import scipy.signal as siglib
import xgboost as xgb
import warnings
import utilz
import neurokit2 as nk
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

augment_signals = True

warnings.filterwarnings('ignore')
fs = fs 
analyzed_list = []
with open('dftemplate.pkl','rb') as target:
        dftemplate = pickle.load(target)
base_dataframe = pd.read_pickle('./base_dataframe.pkl')
for ecg_lead, ecg_label, ecg_name in zip(ecg_leads, ecg_labels, ecg_names):
        signal = ecg_lead    
        if(ecg_label=='N' or ecg_label=='A'):
            try:
                signal_list = []
                signal_list.append(signal)
                if(augment_singnals):
                    for i in range(3):
                        signal_list.append(utilz.augment_signal(signal,fs))
                        i = i+1
                for signal in signal_list:
                    signals, info = nk.ecg_process(signal, sampling_rate=fs, method='neurokit')
                    analyzed = nk.ecg_analyze(signals, sampling_rate=fs)
                    own = utilz.ownFeatures(signals)
                    analyzed = pd.concat([analyzed,own], axis=1)
                    analyzed.replace([np.inf, -np.inf], np.nan, inplace=True)
                    analyzed = pd.concat([dftemplate,analyzed], axis=0)
                    analyzed = analyzed[dftemplate.columns.to_list()].iloc[1].to_frame().T
                    analyzed['TYPE'] = ''
                    if ecg_label=='N':
                        analyzed['TYPE'] = 'N'        # Zuordnung zu "Normal"
                    if ecg_label=='A':
                        analyzed['TYPE'] = 'A'             # Zuordnung zu "Vorhofflimmern"
                    analyzed_list.append(analyzed)
            except:
                print('One sample that could not be used for training!')
df = pd.concat(analyzed_list, axis=0)
df = df.sample(frac=1).reset_index(drop=True)
df = df.replace('N', 0)
df = df.replace('A', 1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numtyp0 = len(df[df['TYPE']==0]['TYPE'])
numtyp1 = len(df[df['TYPE']==1]['TYPE'])
weight = numtyp0 / numtyp1
y_comp = df['TYPE']
X_comp = df.drop('TYPE', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_comp, y_comp, test_size=0.2, random_state=99, shuffle=True)
model = xgb.XGBClassifier(scale_pos_weight =weight)
model.load_model('xgboost_abgabe.json')
model.fit(X_train,y_train)
predict = model.predict(X_test)
accuracy= float(np.sum(predict==y_test))/y_test.shape[0]
print(accuracy)
model.save_model('xgboost_trained.json')


