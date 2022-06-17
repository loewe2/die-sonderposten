#%%
'''
Pipeline-Preprocessing (first normalize, then denoise)
'''
ecg_leads_norm, ecg_leads_detrend, ecg_leads_denoise = ecg_empty(ecg_leads, 3)
for i, dataset in enumerate(ecg_leads):
    # Detrend
    trend, seasonal, residual = ecg_season_trend(ecg_leads[i], fs)
    ecg_leads_detrend[i] = np.asarray(ecg_leads[i] - trend)

    # Normalize
    ecg_leads_norm[i] = ecg_norm(ecg_leads_detrend[i])

    # Denoise
    data, freq = ecg_furier(ecg_leads_norm[i], fs, center=True)
    data_den = ecg_denoise_spectrum(data, freq, 1, 25, "gauß")
    ecg_leads_denoise[i] = ecg_invfurier(data_den, center=True)
    print("Pipeline:"+str(i))

#%%
'''
Feature Extraction (Hamilton or Pan-Tompkins)
'''
peaks, peaks_diff, peaks_diff_norm, peaks_diff_mean, peaks_reshaped, ecg_std, peaks_std, invalid = ecg_empty(
    ecg_leads, 8)
feature_dim = 3
for j, ecg_lead in enumerate(ecg_leads):
    peaks[j], peaks_diff[j] = ecg_detect(ecg_leads[j], fs, method="pan")
    # Don't apply calculations to signals, where the number of found QRS-complexes is not sufficient
    if peaks_diff[j].size >= feature_dim:
        peaks_diff_norm[j] = ecg_norm(peaks_diff[j])
        peaks_reshaped[j], peaks_std[j] = ecg_poincare(
            peaks_diff_norm[j], feature_dim)
        peaks_diff_mean[j] = ecg_diff_mean(peaks_reshaped[j])
        ecg_std[j] = np.std(ecg_leads[j])
        invalid[j] = True
    else:
        peaks_diff_norm[j] = np.inf
        peaks_reshaped[j], peaks_std[j] = np.inf, np.inf
        peaks_diff_mean[j] = np.inf
        ecg_std[j] = np.inf
        invalid[j] = False
    print("Pipeline:" + str(j))

# Save variables
np.save("variables/ecg_leads_detrend.npy", ecg_leads_detrend)
np.save("variables/ecg_leads_norm.npy", ecg_leads_norm)
np.save("variables/ecg_leads_denoise.npy", ecg_leads_denoise)
###
np.save("variables/peaks.npy", peaks)
np.save("variables/peaks_diff.npy", peaks_diff)
np.save("variables/peaks_diff_norm.npy", peaks_diff_norm)
np.save("variables/peaks_reshaped.npy", peaks_reshaped)
np.save("variables/peaks_diff_mean.npy", peaks_diff_mean)
np.save("variables/peaks_std.npy", peaks_std)
np.save("variables/ecg_std.npy", ecg_std)
np.save("variables/invalid.npy", invalid)

#%%
# Load variables
ecg_leads_detrend = np.load(
    "variables/ecg_leads_detrend.npy", allow_pickle=True)
ecg_leads_norm = np.load("variables/ecg_leads_norm.npy", allow_pickle=True)
ecg_leads_denoise = np.load(
    "variables/ecg_leads_denoise.npy", allow_pickle=True)
###
peaks = np.load("variables/peaks.npy", allow_pickle=True)
peaks_diff = np.load("variables/peaks_diff.npy", allow_pickle=True)
peaks_diff_norm = np.load("variables/peaks_diff_norm.npy", allow_pickle=True)
peaks_diff_mean = np.load("variables/peaks_diff_mean.npy", allow_pickle=True)
peaks_reshaped = np.load("variables/peaks_reshaped.npy", allow_pickle=True)
peaks_diff_mean = np.load("variables/peaks_diff_mean.npy", allow_pickle=True)
peaks_std = np.load("variables/peaks_std.npy", allow_pickle=True)
ecg_std = np.load("variables/ecg_std.npy", allow_pickle=True)
invalid = np.load("variables/invalid.npy", allow_pickle=True)
