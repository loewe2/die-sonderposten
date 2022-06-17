# -*- coding: utf-8 -*-
'''
Main script

@author: Julian Hüsselmann, ...
'''

# %%
from feature_extraction import *
from wettbewerb import load_references, save_predictions
from preprocess import *

import neurokit2 as nk
from joblib import dump, load

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import *

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
for p, ecg_lead in enumerate(ecg_leads):
    d.append(
        {
            'ecg_data': ecg_names[p],
            'label': ecg_labels_enc[p]
        }
    )
features_names = pd.DataFrame(d)

bad_sector = []
# Feature arrays from Neurokit2
for j, ecg_lead in enumerate(ecg_leads):
    try:
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
    except:
        bad_sector.append(j)
        continue

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

# Merge all feature arrays horizontally
features_names = features_names.reset_index(drop=True)
features_hrv_time = features_hrv_time.reset_index(drop=True)
features_hrv_freq = features_hrv_freq.reset_index(drop=True)
#features_comp = features_comp.reset_index(drop=True)
features = pd.concat([features_names, features_hrv_time, features_hrv_freq], axis=1)

# Remove all columns/rows, which only have NaN-entries and bad_sectors
features = features.drop(index=bad_sector)
features = features.dropna(axis=1, how="all")
features = features.dropna(axis=0, how="all", subset=features.iloc[:, 2:].columns)

#%%
# Save variable
features_hrv_freq.to_pickle("variables/features_hrv_freq")
features_hrv_time.to_pickle("variables/features_hrv_time")
features.to_pickle("variables/features")
np.save("variables/bad_sector.npy", bad_sector)
#features = pd.read_pickle("variables/features")

#%%
'''
Classificators
'''
# Nur Zweiklassenproblem
features = features[features.iloc[:, 1] != 2]
features = features[features.iloc[:, 1] != 3]

#%%
# SVM - Train
split = 3000
X_train = features.iloc[0:split, 2:]
y_train = features.iloc[0:split, 1]
X_test = features.iloc[split:, 2:]
y_test = features.iloc[split:, 1]
clf = SVC(C=1, kernel='rbf', gamma='auto', cache_size=500, class_weight="balanced")
clf.fit(X_train, y_train)
dump(clf, "variables/svm_model.joblib")

#%%
# SVM - Predict
pred = clf.predict(X_test)
clf.decision_function(X_test)

#%%
# SVM - Debug
print("Accuracy: " + str(accuracy_score(y_test, pred)))
print("Recall (True positive rate): " + str(recall_score(y_test, pred)))
print("Precision: " + str(precision_score(y_test, pred)))
print("F1: " + str(f1_score(y_test, pred)))

#%% Get most important features (10, 14, 15, 18, 21, 22, 25, 26)
pca = PCA(n_components=2)
pca.fit(X_train)
features_series = pd.DataFrame(pca.components_, columns=X_train.columns)
features_series = features_series.mul(pca.explained_variance_ratio_,axis=0).abs().sum().sort_values(ascending=False)
ax = features_series.plot.bar(x='', y='', rot=0)
plt.xticks(rotation=90)

#%%
# Plot SVM
# Source: https://stackoverflow.com/questions/43284811/plot-svm-with-matplotlib
plot_decision_regions(X=X_train.values,
                      y=y_train.values,
                      clf=clf,
                      legend=2)

#%% Feature investigation
x = features.iloc[:, 1]
y = features.iloc[:, 27]
label = features.iloc[:, 1]
colors = ['Red', 'Blue']

fig = plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0, max(label), max(label)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)