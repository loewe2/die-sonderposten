# -*- coding: utf-8 -*-
"""

"""

ecg_leads,ecg_labels,fs,ecg_names = load_references() # Importiere EKG-Daten

detectors = Detectors(fs)                                 # Initialisierung des QRS-Detektors
sdnn_normal = np.array([])                                # Initialisierung der Feature-Arrays
sdnn_afib = np.array([])
for idx, ecg_lead in enumerate(ecg_leads):
    r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
    sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
    if ecg_labels[idx]=='N':
      sdnn_normal = np.append(sdnn_normal,sdnn)         # Zuordnung zu "Normal"
    if ecg_labels[idx]=='A':
      sdnn_afib = np.append(sdnn_afib,sdnn)             # Zuordnung zu "Vorhofflimmern"
    if (idx % 100)==0:
      print(str(idx) + "\t EKG Signale wurden verarbeitet.")