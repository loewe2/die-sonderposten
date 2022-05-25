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

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
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

#%%
'''
Pipeline-Preprocessing (first normalize, then denoise)
'''

ecg_leads_norm, ecg_leads_detrend, ecg_leads_denoise = ecg_empty(ecg_leads, 3)
for i, dataset in enumerate(ecg_leads):
    # Detrend
    trend, seasonal, residual = ecg_season_trend(ecg_leads[i], fs)
    ecg_leads_detrend[i] = np.asarray(ecg_leads[i] - trend)

    # Normalize
    ecg_leads_norm[i] = ecg_norm(ecg_leads_detrend[i])

    # Denoise
    data, freq = ecg_furier(ecg_leads_norm[i], fs, center=True)
    data_den = ecg_denoise_spectrum(data, freq, 1, 25, "gauß")
    ecg_leads_denoise[i] = ecg_invfurier(data_den, center=True)
    print("Pipeline:"+str(i))

#%%
'''
Feature Extraction (Hamilton or Pan-Tompkins)
'''

peaks, peaks_diff, peaks_diff_norm, peaks_diff_mean, peaks_reshaped, ecg_std, peaks_std = ecg_empty(ecg_leads, 7)
for j, ecg_lead in enumerate(ecg_leads):
    peaks[j], peaks_diff[j] = ecg_detect(ecg_leads[j], fs, method="pan")
    # Don't apply calculations to signals, where no QRS-complexes have been found
    if peaks_diff[j].size != 0:
        peaks_diff_norm[j] = ecg_norm(peaks_diff[j])
        peaks_reshaped[j], peaks_std[j] = ecg_poincare(peaks_diff_norm[j], 3)
        peaks_diff_mean[j] = ecg_diff_mean(peaks_reshaped[j])
        ecg_std[j] = np.std(ecg_leads[j])
    else:
        peaks_reshaped[j], peaks_std[j], ecg_std[j], peaks_diff_mean[j] = None, None, None, None
    print("Pipeline:"+str(j))

#%%
'''
Variable management
'''
# Save variables
np.save("variables/ecg_leads_detrend.npy", ecg_leads_detrend)
np.save("variables/ecg_leads_norm.npy", ecg_leads_norm)
np.save("variables/ecg_leads_denoise.npy", ecg_leads_denoise)
###
np.save("variables/peaks.npy", peaks)
np.save("variables/peaks_diff.npy", peaks_diff)
np.save("variables/peaks_diff_norm.npy", peaks_diff_norm)
np.save("variables/peaks_reshaped.npy", peaks_reshaped)
np.save("variables/peaks_diff_mean.npy", peaks_diff_mean)
np.save("variables/peaks_std.npy", peaks_std)
np.save("variables/ecg_std.npy", ecg_std)

#%%
# Load variables
ecg_leads_detrend = np.load("variables/ecg_leads_detrend.npy", allow_pickle=True)
ecg_leads_norm = np.load("variables/ecg_leads_norm.npy", allow_pickle=True)
ecg_leads_denoise = np.load("variables/ecg_leads_denoise.npy", allow_pickle=True)
###
peaks = np.load("variables/peaks.npy", allow_pickle=True)
peaks_diff = np.load("variables/peaks_diff.npy", allow_pickle=True)
peaks_diff_norm = np.load("variables/peaks_diff_norm.npy", allow_pickle=True)
peaks_diff_mean = np.load("variables/peaks_diff_mean.npy", allow_pickle=True)
peaks_reshaped = np.load("variables/peaks_reshaped.npy", allow_pickle=True)
peaks_diff_mean = np.load("variables/peaks_diff_mean.npy", allow_pickle=True)
peaks_std = np.load("variables/peaks_std.npy", allow_pickle=True)
ecg_std = np.load("variables/ecg_std.npy", allow_pickle=True)

#%%
'''
Classificators
'''
# SVM - Train
X = np.vstack(peaks_diff_mean).astype('float')
X = np.nan_to_num(X, copy=False, nan=0.5, posinf=0.5,
                  neginf=0.5)
y = ecg_labels_enc
clf = SVC(C=1, kernel='rbf', gamma='auto', cache_size=500)
clf.fit(X[0:5000], y[0:5000])

#%%
# SVM - Predict
clf.predict(X[5000:])
clf.predict_proba(X)
clf.decision_function(X)

#%%
# SVM - Debug
clf.score(X, y)
clf.get_params()


# fig = plt.figure()
# for i in range(1, 600):
#     aa = peaks_diff[i]
#     peaks_diff_norm = ecg_norm(aa)
#     peaks_reshaped, peaks_std = ecg_poincare(peaks_diff_norm, 2)
#     peaks_reshaped_x = peaks_reshaped[:, 0]
#     peaks_reshaped_y = peaks_reshaped[:, 1]
#     x = ecg_diff_mean(peaks_reshaped_x)
#     y = ecg_diff_mean(peaks_reshaped_y)
#     if ecg_labels[i] == "N":
#         color = 'green'
#     if ecg_labels[i] == "A":
#         color = 'red'
#     if ecg_labels[i] == "~":
#         color = 'white'
#     if ecg_labels[i] == "O":
#         color = 'white'
#     plt.scatter(x, y, color=color)

#%%


C = 1         # Regularisierung
sigma = 1     # Breite Gauss-Kernel

X, y = make_moons(n_samples=150, noise=0.3, random_state=42)

# --- SVM mit Gauss-Kernel ---

sca = StandardScaler().fit(X)
svm = SVC(C=C, gamma=0.5/sigma).fit(sca.transform(X), y)

ng = 101
x1 = np.linspace(0, 1, ng)
x2 = np.linspace(0, 1, ng)
X1, X2 = np.meshgrid(x1, x2)
XX = np.c_[X1.ravel(), X2.ravel()]
Z  = svm.decision_function(sca.transform(XX)).reshape([ng, ng])
plt.contour(X1, X2, Z, [-1, 0, 1])
SV = sca.inverse_transform(svm.support_vectors_)
I0 = (svm.dual_coef_ < 0).ravel()
I1 = (svm.dual_coef_ > 0).ravel()

plt.plot(X[y == 0, 0], X[y == 0, 1], 'C1.', X[y == 1, 0], X[y == 1, 1], 'C0.')
plt.plot(SV[I0, 0], SV[I0, 1], 'C1o', mfc='white')
plt.plot(SV[I1, 0], SV[I1, 1], 'C0o', mfc='white')
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()
