import numpy as np
import streamlit as st
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

# -----------------------------
# Artifact correction (Kubios-style simple threshold)
# -----------------------------
def artifact_correction(rr):

    rr = np.array(rr, dtype=float)

    # Physiological limits
    mask = (rr > 300) & (rr < 2000)
    rr = rr[mask]

    # Local median filter (5-beat window)
    corrected = rr.copy()
    artifact_count = 0

    for i in range(2, len(rr)-2):
        local_median = np.median(rr[i-2:i+3])
        if abs(rr[i] - local_median) > 0.20 * local_median:
            corrected[i] = local_median
            artifact_count += 1

    artifact_ratio = artifact_count / len(rr)

    return corrected, artifact_ratio


# -----------------------------
# HRV Analysis
# -----------------------------
def compute_hrv(rr):

    rr_corrected, artifact_ratio = artifact_correction(rr)

    if artifact_ratio > 0.05:
        return None, "Artifact ratio >5% → 분석 제외"

    # ----- Time Domain -----
    mean_rr = np.mean(rr_corrected)
    mean_hr = 60000 / mean_rr

    sdnn = np.std(rr_corrected, ddof=1)
    diff_rr = np.diff(rr_corrected)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100

    # ----- Frequency Domain -----

    # cumulative time (seconds)
    t = np.cumsum(rr_corrected) / 1000.0
    t = t - t[0]

    # 4 Hz interpolation
    fs = 4
    t_interp = np.arange(0, t[-1], 1/fs)

    cs = CubicSpline(t, rr_corrected)
    rr_interp = cs(t_interp)

    # detrend (linear)
    rr_interp = rr_interp - np.polyval(np.polyfit(t_interp, rr_interp, 1), t_interp)

    # Welch PSD
    f, pxx = welch(
        rr_interp,
        fs=fs,
        window='hamming',
        nperseg=256,
        noverlap=128,
        scaling='density'
    )

    # LF / HF bands
    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)

    lf_power = trapezoid(pxx[lf_band], f[lf_band])
    hf_power = trapezoid(pxx[hf_band], f[hf_band])

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    results = {
        "Mean RR (ms)": mean_rr,
        "Mean HR (bpm)": mean_hr,
        "SDNN (ms)": sdnn,
        "RMSSD (ms)": rmssd,
        "pNN50 (%)": pnn50,
        "LF Power": lf_power,
        "HF Power": hf_power,
        "LF/HF Ratio": lf_hf_ratio,
        "Artifact Ratio (%)": artifact_ratio * 100
    }

    return results, None
