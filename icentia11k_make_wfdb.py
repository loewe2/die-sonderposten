import gzip
import os
import pickle
import warnings

import numpy as np
#import wfdb
#import time

# https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.show_ann_labels
label_mapping = {"btype": {0: ('Q', ''),       # Undefined: Unclassifiable beat
                           1: ('N', ''),       # Normal: Normal beat
                           2: ('S', ''),       # ESSV (PAC): Premature or ectopic supraventricular beat
                           3: ('a', ''),       # Aberrated: Aberrated atrial premature beat
                           4: ('V', '')},      # ESV (PVC): Premature ventricular contraction
                 "rtype": {0: ('', ''),        # Null/Undefined
                           1: ('', ''),        # End
                           2: ('', ''),        # Noise
                           3: ('+', "(N"),     # NSR (normal sinusal rhythm): Normal sinusal rhythm
                           4: ('+', "(AFIB"),  # AFib: Atrial fibrillation
                           5: ('+', "(AFL"),   # AFlutter: Atrial flutter
                           6: (None, None)}}   # Used to split a rhythm when a beat annotation is not
                                               # linked to a rhythm type


def get_person_attributes(path):
    dirname, filename = os.path.split(path)
    filename, ext = filename.split('.')[0], '.'.join(filename.split('.')[1:])
    idx = int(filename.split('_')[0])
    # XXXXX_batched.pkl.gz
    data_path = os.path.join(dirname, f"{filename}.{ext}")
    # XXXXX_batched_lbls.pkl.gz
    labels_path = os.path.join(dirname, f"{filename}_lbls.pkl.gz")
    return idx, data_path, labels_path


def make_wfdb(path):
    output=[]
    """Create a WFDB file from a .../XXXXX_batched.pkl.gz"""
    person_idx, person_data_path, person_labels_path = \
        get_person_attributes(path)
   
    assert os.path.exists(person_data_path)
    assert os.path.exists(person_labels_path)

    with gzip.open(person_data_path) as gzf:
        person_data = pickle.load(gzf)
    with gzip.open(person_labels_path) as gzf:
        person_labels = pickle.load(gzf)
    cwd = os.getcwd()
    
    try:
        # wfdb.io.wrann does not allow '/' in path which forces the cwd to be
        # changed to the person's subdir
        # os.makedirs(f"p{person_idx:05d}", exist_ok=True)
        # os.chdir(f"p{person_idx:05d}")
        
        for i, (p_signal, seg_labels) in enumerate(zip(person_data,
                                                       person_labels)):
            # https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.wrsamp
            #writing of the .dat file

            beats = np.unique(np.concatenate(seg_labels["btype"]))
            rhythms = np.unique(np.concatenate(seg_labels["rtype"]))
            if not np.in1d(beats, rhythms).all():
                warnings.warn("Beat annotations count does not match with "
                    "rhythm annotations count for p{:05d}_s{:02d}. Spliting "
                    "rhythm zones surrounding extra beats".format(
                        person_idx, i))
                seg_labels["rtype"].append(beats[np.in1d(beats, rhythms) == False])

            flat_ann = []
            for label_type, labels in seg_labels.items():
                for type_id, locs in enumerate(labels):
                    if locs.size and label_type == "rtype" and type_id == 0:
                        warnings.warn("Undefined rhythm annotation should not "
                            "exist in icentia11k for p{:05d}_s{:02d}".format(
                                person_idx, i))
                    flat_ann.extend([(loc, label_type,
                                      *label_mapping[label_type][type_id])
                                     for loc in locs])
            if not flat_ann:
                warnings.warn("Empty annotations for p{:05d}_s{:02d}".format(
                    person_idx, i))
                continue
            flat_ann.sort()

            sample, ann_type, symbol, _aux_note = zip(*flat_ann)
            sample = np.asarray(sample)
            symbol = np.asarray(symbol)
            chan = np.asarray([0] * sample.size)
            aux_note = np.asarray(_aux_note, dtype="<U6")

            rtype = np.where(np.asarray(ann_type) == "rtype")[0]
            for _i in range(1, len(rtype) + 1):
                prev_idx, idx = (rtype[_i-1],
                                 rtype[_i] if _i < len(rtype) else None)
                prev_aux, aux = (_aux_note[prev_idx],
                                 _aux_note[idx] if idx is not None else None)
                if aux and prev_aux == aux:
                    symbol[idx] = ''
                    aux_note[idx] = ''
                elif prev_aux:
                    # Single beat in rhythm zone
                    if symbol[prev_idx] == '+':
                        aux_note[prev_idx] += ')'
                    # Closing rhythm zone
                    else:
                        symbol[prev_idx] = '+'
                        aux_note[prev_idx] = ')'

            mask = (symbol != '') * (symbol != None)
            sample = sample[mask]
            symbol = symbol[mask]
            aux_note = aux_note[mask]
            chan = np.array([0] * sample.size)
            # https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.wrann
            output.append((p_signal, aux_note, sample))
    finally:
        # chdir back to initial cwd
        #os.chdir(cwd)
        return output