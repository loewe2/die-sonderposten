import CNNModels.icentia11k_make_wfdb as wfdbcreator
import numpy as np
import random
import scipy.signal as siglib
import numpy as np
from ecgdetectors import Detectors


# This function is called by the data generator to create one batch of data.
# patient: number of the patient in the icentia dataset
# subset: number of the subset from the patient that is used
# batchsize: the size of the batch
def getData_1dcnn(patient, subset, batchsize=1024, multi=True, nperseg_value=64):
    number_as_string = f'{patient:05d}'
    path = '/shared_data/icentia11k/'+number_as_string+'_batched.pkl.gz'
    signals, signal_lab = icentiaDataProcessor(path)
    startindex = subset*batchsize
    endindex = (subset+1)*batchsize
    signals = signals[startindex:endindex]
    signal_lab = signal_lab[startindex:endindex]
    # ab hier kommt dein eigener code, es wird über signals iteriert und für jedes einzelen signal dann die passende vorverarbeitung durchgeführt, so wie du sie brauchst
    # zusätzlich wird für jedes label ein codierung anglegt, aber das wäre dann wieder abhängig von deiner verarbeitung
    spectro = []
    spectro_label = []
    for i in range(len(signals)):
        signal = signals[i]
        signal = siglib.resample(signal, int(len(signal)*1.2))
        signal = np.ravel(signal)
        signal = preprocessData(signal, 9000, 300, False)
        f, t, Sxx = siglib.spectrogram(signal, fs=300, nfft=512, nperseg=64)
        Sxx = 10*np.log10(Sxx+1e-12)
        Sxx = np.reshape(Sxx, (Sxx.shape[0], Sxx.shape[1], 1))

        if signal_lab[i] == 'N':  # Zuordnung zu "Normal"
            label = np.asarray([1, 0, 0, 0])

        if signal_lab[i] == 'A':  # Zuordnung zu "Vorhofflimmern"
            label = np.asarray([0, 1, 0, 0])

        if multi:  # Zuordnung zu "Noise"
            if signal_lab[i] == '~':
                label = np.asarray([0, 0, 1, 0])

            if signal_lab[i] == 'O':  # Zuordnung zu "Other"
                label = np.asarray([0, 0, 0, 1])
        spectro.append(Sxx)
        spectro_label.append(label)
        del f, t, Sxx, label
    return spectro, spectro_label


# This function is called by the data generator to create one batch of data.
# patient: number of the patient in the icentia dataset
# subset: number of the subset from the patient that is used
# batchsize: the size of the batch
def getData_spectro(patient, subset, batchsize=1024, multi=True, nperseg_value=64):
    number_as_string = f'{patient:05d}'
    path = '/shared_data/icentia11k/'+number_as_string+'_batched.pkl.gz'
    signals, signal_lab = icentiaDataProcessor(path)
    startindex = subset*batchsize
    endindex = (subset+1)*batchsize
    signals = signals[startindex:endindex]
    signal_lab = signal_lab[startindex:endindex]
    spectro = []
    spectro_label = []
    for i in range(len(signals)):
        signal = signals[i]
        signal = siglib.resample(signal, int(len(signal)*1.2))
        signal = np.ravel(signal)
        signal = preprocessData(signal, 9000, 300, False)
        f, t, Sxx = siglib.spectrogram(signal, fs=300, nfft=512, nperseg=64)
        Sxx = 10*np.log10(Sxx+1e-12)
        Sxx = np.reshape(Sxx, (Sxx.shape[0], Sxx.shape[1], 1))

        if signal_lab[i] == 'N':  # Zuordnung zu "Normal"
            label = np.asarray([1, 0, 0, 0])

        if signal_lab[i] == 'A':  # Zuordnung zu "Vorhofflimmern"
            label = np.asarray([0, 1, 0, 0])

        if multi:  # Zuordnung zu "Noise"
            if signal_lab[i] == '~':
                label = np.asarray([0, 0, 1, 0])

            if signal_lab[i] == 'O':  # Zuordnung zu "Other"
                label = np.asarray([0, 0, 0, 1])
        spectro.append(Sxx)
        spectro_label.append(label)
        del f, t, Sxx, label
    return spectro, spectro_label

# function opens corresponding file and returns the fitting signals and labels


def icentiaDataProcessor(path):
    lischte = wfdbcreator.make_wfdb(path)

    # import all data
    records = []
    labels = []
    timestamps = []
    for i in range(len(lischte)):
        records.append(lischte[i][0])
        labels.append(lischte[i][1])
        timestamps.append(lischte[i][2])
        where = np.where(labels[i] != '')
        labels[i] = labels[i][where]
        timestamps[i] = timestamps[i][where]

    # create segments
    segments = []
    segments_label = []

    for j in range(len(records)):
        for i in range(len(labels[j])-1):
            if(labels[j][i] == '(N' and labels[j][i+1] == ')'):
                segments_label.append('N')
                segments.append(
                    records[j][timestamps[j][i]:timestamps[j][i+1]])
            elif(labels[j][i] == '(AFIB' and labels[j][i+1] == ')'):
                segments_label.append('A')
                segments.append(
                    records[j][timestamps[j][i]:timestamps[j][i+1]])
            elif(labels[j][i] == '(AFL' and labels[j][i+1] == ')'):
                segments_label.append('O')
                segments.append(
                    records[j][timestamps[j][i]:timestamps[j][i+1]])
            elif(labels[j][i] == 'None' or labels[j][i] == ')'):
                pass
            else:
                pass

    # length of recordings that I want to create, similar to the length of recordings in the original trainingsset
    random_numbers = np.load('patientdata_evaluated/random_numbers.npy')
    max_duration = 60

    signals = []
    signals_label = []
    fs = 250
    for i in range(len(segments)):
        current_segment = segments[i]
        index = 0
        while(len(current_segment) > fs*max_duration):
            time = random_numbers[index]
            new_signal = current_segment[0:fs*time]
            current_segment = current_segment[fs*time:-1]
            signals.append(new_signal)
            signals_label.append(segments_label[i])
            index = index+1

    return (signals, signals_label)

# Preprocessing of the incoming data
# the signal is cut before the first and after the last detected r_peak if it is to short
# than the signal is repeated multiple times to reach the desired length
# afterwards the signal is normalized


def preprocessData(data, targetsize, samplefreq, debug=False):
    ret_signal = 0
    repeat_signal = 0
    # looking for r_peaks and repeating the signal
    if(data.size < targetsize):
        if(debug):
            print('Length of the signal ->'+str(data.size) +
                  '<- is shorter than '+str(targetsize))
        detector = Detectors(samplefreq)
        try:
            r_peaks = detector.hamilton_detector(data)
            if(debug):
                print('R-Peaks are detected!')
            repeat_signal = data[r_peaks[0]:r_peaks[-1]]
        except:
            if(debug):
                print('R-Peaks could not be detected!')
            repeat_signal = data

        if(repeat_signal.size == 0):
            if(debug):
                print('Something went wrong, just use the original signal to repeat.')
            repeat_signal = data
            if(debug and repeat_signal.size == 0):
                print('I have no idea.')
        repeats = int(np.ceil(targetsize / repeat_signal.size))
        if(debug):
            print('Signal is repeated ' + str(repeats) + ' times.')
        for i in range(repeats):
            ret_signal = np.append(ret_signal, repeat_signal)
    else:
        if(debug):
            print('Length of the signal ->'+str(data.size) +
                  '<- is bigger than '+str(targetsize))
        ret_signal = data

    # cutting the signal to the desired length
    # same ammount is cut at the start and the end of the signal
    if(ret_signal.size > targetsize):
        if(debug):
            print('Signal is cut to length: ' + str(targetsize))
        cutindex = int(((ret_signal.size-targetsize)/2))
        ret_signal = ret_signal[cutindex:-1]
        ret_signal = ret_signal[0:targetsize]
        if(debug):
            print('Size of the new signal: ' + str(ret_signal.size))

    # normalising the signal
    ret_signal = (ret_signal - np.min(ret_signal)) / \
        (np.max(ret_signal) - np.min(ret_signal))

    return ret_signal
