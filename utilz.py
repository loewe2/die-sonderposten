from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

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


    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    
    return (f1, acc)

