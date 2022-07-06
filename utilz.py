from pandas import read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

def predictions_score(y_true_path, y_pred_path):

    y_true_all = read_csv(y_true_path, delimiter=',', header=None)
    y_pred_all = read_csv(y_pred_path, delimiter=',', header=None)

    enc = OneHotEncoder()
    enc.fit(y_true_all[1].values.reshape(-1,1))
    y_true = enc.transform(y_true_all[1].values.reshape(-1,1)).toarray()
    y_pred = enc.transform(y_pred_all[1].values.reshape(-1,1)).toarray()

    f1 = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    
    return (f1, acc)

