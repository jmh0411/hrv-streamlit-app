import numpy as np
import streamlit as st
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

st.title("5-Minute HRV Analysis (Research Grade)")

uploaded_file = st.file_uploader("Upload RR interval file (.txt)", type=["txt"])

# -----------------------------
# Artifact correction
# -----------------------------
def artifact_correction(rr):

    rr = np.array(rr, dtype=float)

    # physiological filtering
    rr = rr[(rr > 300) & (rr < 2000)]

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
# HRV computation
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
    t = np.cumsum(rr_corrected) / 1000.0
    t = t - t[0]

    fs = 4
    t_interp = np.arange(0, t[-1], 1/fs)

    cs = CubicSpline(t, rr_corrected)
    rr_interp = cs(t_interp)

    # detrend
    rr_interp = rr_interp - np.polyval(np.polyfit(t_interp, rr_interp, 1), t_interp)

    f, pxx = welch(
        rr_interp,
        fs=fs,
        window='hamming',
        nperseg=256,
        noverlap=128,
        scaling='density'
    )

    lf_band = (f >= 0.04) & (f < 0.15)
    hf_band = (f >= 0.15) & (f < 0.40)

    lf_power = trapezoid(pxx[lf_band], f[lf_band])
    hf_power = trapezoid(pxx[hf_band], f[hf_band])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    results = {
        "Mean RR (ms)": round(mean_rr, 2),
        "Mean HR (bpm)": round(mean_hr, 2),
        "SDNN (ms)": round(sdnn, 2),
        "RMSSD (ms)": round(rmssd, 2),
        "pNN50 (%)": round(pnn50, 2),
        "LF Power": round(lf_power, 6),
        "HF Power": round(hf_power, 6),
        "LF/HF Ratio": round(lf_hf_ratio, 3),
        "Artifact Ratio (%)": round(artifact_ratio * 100, 2)
    }

    return results, None


# -----------------------------
# Run analysis
# -----------------------------
if uploaded_file is not None:

    rr = np.loadtxt(uploaded_file)

    if st.button("Analyze HRV"):

        results, error = compute_hrv(rr)

        if error:
            st.error(error)
        else:
            st.success("Analysis Complete")
            st.write(results)
