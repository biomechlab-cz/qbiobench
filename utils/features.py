"""Feature extraction per task (fixed feature sets).

Built on NeuroKit2. Each function returns a flat feature vector with a stable,
named column order (the `*_FEATURES` lists) so columns are reproducible.
Extractors are defensive: on detection failure a row of NaN is returned and the
builder imputes/drops it (logged), rather than crashing a whole campaign.

Verified column names against NeuroKit2 0.2.x on real signals (2026-06-09).
"""
from __future__ import annotations

import numpy as np
import neurokit2 as nk
from scipy.signal import welch

# ===================== T2 Apnea-ECG: 28 HRV + 2 EDR = 30 =====================
APNEA_HRV = [
    "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
    "HRV_MedianNN", "HRV_MadNN", "HRV_IQRNN", "HRV_pNN50", "HRV_pNN20",
    "HRV_MinNN", "HRV_MaxNN", "HRV_HTI", "HRV_LF", "HRV_HF", "HRV_LFHF",
    "HRV_LFn", "HRV_HFn", "HRV_LnHF", "HRV_HFD", "HRV_TP", "HRV_SD1", "HRV_SD2",
    "HRV_SD1SD2", "HRV_SampEn", "HRV_ApEn", "HRV_DFA_alpha1",
]
APNEA_FEATURES = APNEA_HRV + ["EDR_PeakFreq", "EDR_BandPower"]


def _hrv_named(rpeaks_df, fs, names):
    try:
        hrv = nk.hrv(rpeaks_df, sampling_rate=fs, show=False)
        return np.array([float(hrv[n].iloc[0]) if n in hrv else np.nan for n in names])
    except Exception:
        return np.full(len(names), np.nan)


def _edr(ecg_seg, rpeak_idx, fs):
    """ECG-derived respiration: R-peak amplitude series → Welch → respiratory
    band (0.1–0.5 Hz) peak frequency + power. Returns [peak_freq, band_power]."""
    if len(rpeak_idx) < 8:
        return np.array([np.nan, np.nan])
    amps = ecg_seg[rpeak_idx]
    t = rpeak_idx / fs
    uni_t = np.arange(t[0], t[-1], 1 / 4.0)            # 4 Hz uniform grid
    amps_u = np.interp(uni_t, t, amps)
    f, p = welch(amps_u - amps_u.mean(), fs=4.0, nperseg=min(len(amps_u), 64))
    band = (f >= 0.1) & (f <= 0.5)
    if not band.any():
        return np.array([np.nan, np.nan])
    return np.array([float(f[band][np.argmax(p[band])]), float(p[band].sum())])


def apnea_minute_features(ecg, fs, labels):
    """Per-minute feature matrix for one Apnea-ECG recording.
    Returns (X[M,30], y[M]). Minute m = ecg[m*60*fs:(m+1)*60*fs]."""
    M = len(labels)
    X = np.full((M, len(APNEA_FEATURES)), np.nan)
    seglen = 60 * fs
    for m in range(M):
        seg = ecg[m * seglen:(m + 1) * seglen]
        if len(seg) < seglen // 2:
            continue
        try:
            rp, info = nk.ecg_peaks(nk.ecg_clean(seg, sampling_rate=fs), sampling_rate=fs)
            ridx = info["ECG_R_Peaks"]
            if len(ridx) < 8:
                continue
            X[m, :28] = _hrv_named(rp, fs, APNEA_HRV)
            X[m, 28:] = _edr(seg, np.asarray(ridx), fs)
        except Exception:
            continue
    return X, labels.astype(int)


# ===================== T3 WESAD: HRV12+EDA10+RSP6+EMG6+TEMP2 = 36 ============
WESAD_HRV = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF",
             "HRV_HF", "HRV_LFHF", "HRV_SD1", "HRV_SD2", "HRV_SampEn",
             "HRV_CVNN", "HRV_HTI"]
WESAD_EDA = ["EDA_TonicMean", "EDA_TonicStd", "EDA_TonicSlope", "EDA_PhasicMean",
             "EDA_PhasicStd", "EDA_PhasicMax", "SCR_Count", "SCR_AmpMean",
             "SCR_AmpStd", "EDA_Level"]
WESAD_RSP = ["RSP_RateMean", "RSP_RateStd", "RSP_AmpMean", "RSP_AmpStd",
             "RSP_RRV_SDBB", "RSP_PeriodMean"]
WESAD_EMG = ["EMG_RMS", "EMG_MAV", "EMG_WL", "EMG_VAR", "EMG_ZC", "EMG_SSC"]
WESAD_TEMP = ["TEMP_Mean", "TEMP_Slope"]
WESAD_FEATURES = WESAD_HRV + WESAD_EDA + WESAD_RSP + WESAD_EMG + WESAD_TEMP


def _eda_features(eda, fs):
    try:
        eda = nk.signal_sanitize(eda.astype(np.float64))
        ds = 8
        eda_ds = nk.signal_resample(eda, sampling_rate=fs, desired_sampling_rate=ds)
        ph = nk.eda_phasic(eda_ds, sampling_rate=ds, method="cvxeda")
        tonic, phasic = ph["EDA_Tonic"].values, ph["EDA_Phasic"].values
        slope = np.polyfit(np.arange(len(tonic)), tonic, 1)[0]
        try:
            _, scr = nk.eda_peaks(phasic, sampling_rate=ds)
            amps = scr.get("SCR_Amplitude", np.array([]))
            amps = np.asarray(amps)[~np.isnan(np.asarray(amps))] if len(amps) else np.array([])
            n_scr, amp_mean, amp_std = len(scr.get("SCR_Peaks", [])), \
                (amps.mean() if len(amps) else 0.0), (amps.std() if len(amps) else 0.0)
        except Exception:
            n_scr, amp_mean, amp_std = 0, 0.0, 0.0
        return np.array([tonic.mean(), tonic.std(), slope, phasic.mean(), phasic.std(),
                         phasic.max(), n_scr, amp_mean, amp_std, eda_ds.mean()])
    except Exception:
        return np.full(len(WESAD_EDA), np.nan)


def _rsp_features(rsp_sig, fs):
    try:
        rsp, _ = nk.rsp_process(rsp_sig.astype(np.float64), sampling_rate=fs)
        rate = rsp["RSP_Rate"].values
        amp = rsp["RSP_Amplitude"].values
        try:
            rrv = nk.rsp_rrv(rsp, sampling_rate=fs)
            sdbb = float(rrv["RRV_SDBB"].iloc[0]) if "RRV_SDBB" in rrv else np.nan
        except Exception:
            sdbb = np.nan
        period = 60.0 / np.nanmean(rate) if np.nanmean(rate) > 0 else np.nan
        return np.array([np.nanmean(rate), np.nanstd(rate), np.nanmean(amp),
                         np.nanstd(amp), sdbb, period])
    except Exception:
        return np.full(len(WESAD_RSP), np.nan)


def _emg_features(emg, fs):
    e = emg.astype(np.float64)
    de = np.diff(e)
    rms = np.sqrt(np.mean(e ** 2))
    mav = np.mean(np.abs(e))
    wl = np.sum(np.abs(de))
    var = np.var(e)
    zc = np.sum((e[:-1] * e[1:]) < 0)
    ssc = np.sum((de[:-1] * de[1:]) < 0)
    return np.array([rms, mav, wl, var, float(zc), float(ssc)])


def wesad_window_features(window: dict, fs):
    """36-dim feature vector for one 60-s multimodal window (dict of channels)."""
    out = np.full(len(WESAD_FEATURES), np.nan)
    # HRV from ECG
    try:
        rp, info = nk.ecg_peaks(nk.ecg_clean(window["ECG"], sampling_rate=fs), sampling_rate=fs)
        if len(info["ECG_R_Peaks"]) >= 8:
            out[:12] = _hrv_named(rp, fs, WESAD_HRV)
    except Exception:
        pass
    out[12:22] = _eda_features(window["EDA"], fs)
    out[22:28] = _rsp_features(window["Resp"], fs)
    out[28:34] = _emg_features(window["EMG"], fs)
    temp = window["Temp"].astype(np.float64)
    out[34:36] = [temp.mean(), np.polyfit(np.arange(len(temp)), temp, 1)[0]]
    return out


# ===================== T1 PTB-XL: per-lead spectral+statistical = 64 =========
# Decision of 2026-06-09: 8 independent leads x 8 features. Deterministic,
# no fragile delineation. Fixed before any result was inspected.
PTBXL_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]   # I, II, V1, V2, V3, V4, V5, V6
PTBXL_PER_LEAD = ["std", "rms", "log_linelength", "bp_0.5-4", "bp_4-15",
                  "bp_15-40", "dom_freq", "spec_entropy"]
PTBXL_FEATURES = [f"L{l}_{f}" for l in PTBXL_LEADS for f in PTBXL_PER_LEAD]  # 64
PTBXL_N_FEATURES = 64


def _lead_features(x, fs):
    x = x.astype(np.float64)
    std = x.std()
    rms = np.sqrt(np.mean(x ** 2))
    ll = np.log1p(np.sum(np.abs(np.diff(x))))
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), 256))
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(p[m].sum()) if m.any() else 0.0
    band = (f >= 0.5) & (f <= 40)
    if band.any() and p[band].sum() > 0:
        dom = float(f[band][np.argmax(p[band])])
        pn = p[band] / p[band].sum()
        spec_ent = float(-np.sum(pn * np.log(pn + 1e-12)))
    else:
        dom, spec_ent = np.nan, np.nan
    return [std, rms, ll, bp(0.5, 4), bp(4, 15), bp(15, 40), dom, spec_ent]


def ptbxl_features(sig_12lead, fs=100):
    """64-dim kernel-path feature vector: 8 leads x 8 spectral/statistical features."""
    out = []
    for lead in PTBXL_LEADS:
        out.extend(_lead_features(sig_12lead[:, lead], fs))
    return np.array(out, dtype=np.float64)


def ptbxl_spectrum(sig_12lead, fs=100):
    """VQC-path raw feature: lead-II rFFT magnitude in 0.5–40 Hz (PCA→8 later)."""
    x = sig_12lead[:, 1].astype(np.float64)
    mag = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    band = (freqs >= 0.5) & (freqs <= 40)
    return mag[band]
