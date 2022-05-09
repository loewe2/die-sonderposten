# -*- coding: utf-8 -*-
"""
Preprocessing of the input data

@author: Julian Hüsselmann
"""

import csv
import scipy.io as sio
import numpy as np

# Import der EKG-Dateien
fs = 300
with open('training/REFERENCE.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data = sio.loadmat('training/' + row[0] + '.mat')   
        ecg_lead = data['val'][0]
        line_count = line_count + 1
        if (line_count % 100) == 0:
            print(str(line_count) + "\t Dateien wurden verarbeitet.")
