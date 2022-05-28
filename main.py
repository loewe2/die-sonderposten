# -*- coding: utf-8 -*-
'''
Main script

@author: Julian Hüsselmann, ...
'''

# %%
from feature_extraction import *
from wettbewerb import load_references, save_predictions
from preprocess import *

from sklearn import preprocessing
from sklearn.svm import SVC
from joblib import dump, load

import neurokit2 as nk
# %%
'''
Data import
'''
ecg_leads, ecg_labels, fs, ecg_names = load_references(folder='training')
idx_N = [i for i in range(len(ecg_labels)) if ecg_labels[i] == 'N']
idx_A = [i for i in range(len(ecg_labels)) if ecg_labels[i] == 'A']
idx_tilde = [i for i in range(len(ecg_labels)) if ecg_labels[i] == '~']
idx_O = [i for i in range(len(ecg_labels)) if ecg_labels[i] == 'O']

# Label encoding (0: A, 1: N, 2: O, 3: ~)
le = preprocessing.LabelEncoder()
le.fit(ecg_labels)
ecg_labels_enc = le.transform(ecg_labels)

'''
Feature extraction
'''
#%%
# Get names and labels
d = []
for p in range(0,80):
    d.append(
        {
            'ecg_data': ecg_names[p],
            'label': ecg_labels[p]
        }
    )
features_names = pd.DataFrame(d)

# Feature arrays from Neurokit2
for j in range(0, 80):
    processed_data, info = nk.ecg_process(ecg_leads[j], sampling_rate=fs)
    hrv_time = nk.hrv_time(info['ECG_R_Peaks'], sampling_rate=fs, show=False)
    hrv_freq = nk.hrv_frequency(info['ECG_R_Peaks'], sampling_rate=fs, show=False, normalize=True)
    #(too long to load): complexity, info_c = nk.complexity(ecg_leads[j], which=["fast", "medium"])
    #(too long to load): delay, parameters = nk.complexity_delay(ecg_leads[j], delay_max=100, show=False, method="rosenstein1994")
    if j == 0:
        features_hrv_time = hrv_time
        features_hrv_freq = hrv_freq
        #features_comp = complexity
    else:
        features_hrv_time = pd.concat([features_hrv_time, hrv_time])
        features_hrv_freq = pd.concat([features_hrv_freq, hrv_freq])
        #features_comp = pd.concat([features_comp, complexity])
    print(j)

'''
Preprocess
'''
#%%
#Impute NaN values
features_hrv_freq.fillna(features_hrv_freq.mean(), inplace=True)
features_hrv_time.fillna(features_hrv_time.mean(), inplace=True)
#features_comp.fillna(features_comp.mean(), inplace=True)

# Normalize all values
features_hrv_freq = pandas_normalize(features_hrv_freq)
features_hrv_time = pandas_normalize(features_hrv_time)
#features_comp = pandas_normalize(features_comp)

# Merge all feature arrays
features_names = features_names.reset_index(drop=True)
features_hrv_time = features_hrv_time.reset_index(drop=True)
features_hrv_freq = features_hrv_freq.reset_index(drop=True)
#features_comp = features_comp.reset_index(drop=True)
features = pd.concat([features_names, features_hrv_time, features_hrv_freq], axis=1)

# Remove all columns, which only have NaN-entries
features = features.dropna(axis=1, how="all")

# Save variable
features.to_pickle("variables/features")

#%%
'''
Classificators
'''
# Nur Zweiklassenproblem
features = features[features.iloc[:, 1] != "~"]
features = features[features.iloc[:, 1] != "O"]

# SVM - Train
X = features.iloc[0:35, 2:16]
y = features.iloc[0:35, 1]
X2 = features.iloc[35:48, 2:16]
y2 = features.iloc[35:48, 1]
clf = SVC(C=0.01, kernel='rbf', gamma='auto', cache_size=500)
clf.fit(X, y)
dump(clf, "variables/svm_model.joblib")

#%%
# SVM - Predict
pred = clf.predict(X2)
clf.decision_function(X2)

#%%
# SVM - Debug
clf.score(X2, y2)
clf.get_params()

#%%
from sklearn.decomposition import FactorAnalysis
pca = PCA(n_components=2)
pca.fit(X)
print(pca.components_)
aa = pd.DataFrame(
    pca.components_, columns=X.columns, index=['PC-1', 'PC-2'])