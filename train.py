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

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

model = keras.models.load_model('spectro.h5') 

spectograms = []
labels  = []
for i in range(len(ecg_leads)):
    signal = ecg_leads[i]
    signal = np.ravel(signal)
    signal = icentiaDataProcessor.preprocessData(signal,9000,300,False)
    f, t, Sxx = siglib.spectrogram(signal, fs=300, nfft=512, nperseg=64)
    Sxx = 10*np.log10(Sxx+1e-12)
    Sxx = np.reshape(Sxx, (Sxx.shape[0], Sxx.shape[1], 1))
    spectograms.append(Sxx)
    if ecg_labels[i]=='N': # Zuordnung zu "Normal"
            label=np.asarray([1,0,0,0])

    if ecg_labels[i]=='A':  # Zuordnung zu "Vorhofflimmern"
            label=np.asarray([0,1,0,0])            

    if ecg_labels[i]=='~':                       
            label=np.asarray([0,0,1,0])
                
    if ecg_labels[i]=='O':#Zuordnung zu "Other"                        
            label=np.asarray([0,0,0,1])
    labels.append(label)
    del f, t, Sxx, label
spectograms = np.asarray(spectograms)
labels = np.asarray(labels)

checkpoint_filepath = 'trained_spectogram.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(spectograms, labels, validation_split=0.2, epochs=10, callbacks=[model_checkpoint_callback])

