# -*- coding: utf-8 -*-
'''
Preprocessing of the input data

@author: Julian Hüsselmann
'''

# %%
from wettbewerb import load_references, save_predictions
from preprocess import *

# %%
# Import der EKG-Dateien
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')

#%%
# Pipeline-Vorverarbeitung:
ecg_leads_pre = [None]*len(ecg_leads)
for i, dataset in enumerate(ecg_leads):
    ecg_leads_pre[i] = ecg_norm(ecg_leads[i])

ecg_denoise_kalman(ecg_leads[1])