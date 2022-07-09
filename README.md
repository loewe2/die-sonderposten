# Team - Die Sonderposten

## Abgabe
Alle Daten zur finalen Abgabe finden Sie im Hauptordner. Das zentrale Notebook, in dem das Model erstellt und trainiert wurde ist 'XGBoost.ipynb'. Hnzu kommen Hilfsfunktionen in 'utilz.py' und die für die Abgabe relevanten Dateien (z.B.: predict.py/train.py).
Die XGBoost-Modelle wurden in JSON-Files gespeichert. Das Modell zur 4. Abgabe ist 'xgboost_abgabe.json', das Modell der finalen (5.) Abgabe ist 'xgboost_augmented.json'.
Für die finale Abgabe haben wir die Trainngsdaten zusätzlich augmented, daher gehören hierzu die Trainingsdaten: 'Neurokit_Dataset_augmented.pkl', 'MIT_Dataset_augmented.pkl'.
In der Datei 'dftemplate.pkl' befindet sich eine Vorlage, die die korrekte Übergabe der Feature an das Modell sicherstellt.

## Versuche
Wir haben diverse Modelle ausprobiert, diese Versuche befinden sich je nach Modell-Typ in den jeweiligen Ordnern in Jupyter-Notebooks:
Markup :    * CNN-Models
                * Diverse CNNs auf Basis von Spectrogrammen (2. Abgabe) und 1DCNNs zur direkten Anwendung auf die Signale
            * Tree_Forest_Models
                 * Decision Trees (1. Abgabe), Random Forests, Gradient Boosted Trees (3. Abgabe)

## Daten
Wir haben neben dem zur Verfügung gestellten Datensatz zusätzlich noch auf den MIT-BIH Atrial Fibrillation Datensatz trainiert und getestet. Einige der CNN-Modelle haben wir auch mittels des Icentia11k Datensatzes auf dem KIS*MED-JupyterHub pretrained, die Vorverarbeitung hierzu finden Sie im Ordner 'CNN-Models'.
Auf Grund der Datenmenge haben wir diese Daten nicht vollständig in den GitHub mit einbezogen.

Die zur Verfügung gestellten Daten befinden sich in den Ordnern: 'test' und 'training'.
Die Vorverarbeitung um den MIT-BIH Atrial Fibrillation Datensatz für uns Nutzbar zu machen (z.B. slicing/labelling) finden sie im Notebook 'Process_MIT_BIH_AtrialFibrillation.ipynb'.
Für die finale Abgabe finden Sie die mittels Neurokit vorverarbeiteten Daten in .pkl-Files im Hauptordner. Einen Teil des MIT-Datensatzes (~6000 Signale) und den bereitgestellten Datensatz.


## Dependencies
- Installiere Pakete über: `pip install -r requirements.txt`

