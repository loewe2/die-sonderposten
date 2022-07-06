#from pandas import read_csv

from numpy import genfromtxt
from sklearn.metrics import accuracy_score, f1_score

def predictions_score(y_true_path, y_pred_path):
    y_true_all = genfromtxt(y_true_path, delimiter=',')
    y_pred_all = genfromtxt(y_pred_path, delimiter=',')

    y_true_temp=y_true_all[1]
    y_pred_temp=y_pred_all[1]


    f1 = f1_score(y_true_temp,y_pred_temp)
    acc = accuracy_score(y_true_temp, y_pred_temp)

    return (f1, acc)

predictions_score('./training/REFERENCE.csv','PREDICTIONS.csv')