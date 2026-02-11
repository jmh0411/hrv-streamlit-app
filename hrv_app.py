import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.stats import ttest_rel

st.title("Pre–Post HRV Analysis (Research Grade)")

remove_first_min = st.checkbox("Remove first 60 seconds (recommended)", value=True)

pre_file = st.file_uploader("Upload PRE RR file (.txt)", type=["txt"])
post_file = st.file_uploader("Upload POST RR file (.txt)", type=["txt"])


# -------------------------
# Artifact Correction
# -------------------------
def artifact_correction(rr):
    rr = np.array(rr, dtype=float)
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


# -------------------------
# HRV Calculation
# -------------------------
def compute_hrv(rr):

    rr_corrected, artifact_ratio = artifact_correction(rr)

    if artifact_ratio > 0.05:
        return None, "Artifact ratio >5%"

    # Remove first 60 sec
    if remove_first_min:
        t = np.cumsum(rr_corrected) / 1000
        mask = t > 60
        rr_corrected = rr_corrected[mask]

    mean_rr = np.mean(rr_corrected)
    mean_hr = 60000 / mean_rr

    sdnn = np.std(rr_corrected, ddof=1)
    diff_rr = np.diff(rr_corrected)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    ln_rmssd = np.log(rmssd)

    # Frequency domain
    t = np.cumsum(rr_corrected) / 1000
    t = t - t[0]

    fs = 4
    t_interp = np.arange(0, t[-1], 1/fs)
    cs = CubicSpline(t, rr_corrected)
    rr_interp = cs(t_interp)

    rr_interp = rr_interp - np.polyval(np.polyfit(t_interp, rr_interp, 1), t_interp)

    f, pxx = welch(
        rr_interp,
        fs=fs,
        window='hamming',
        nperseg=256,
        noverlap=128,
        scaling='density'
    )

    hf_band = (f >= 0.15) & (f < 0.40)
    hf_power = trapezoid(pxx[hf_band], f[hf_band])
    ln_hf = np.log(hf_power) if hf_power > 0 else np.nan

    return {
        "Mean HR": mean_hr,
        "RMSSD": rmssd,
        "lnRMSSD": ln_rmssd,
        "HF": hf_power,
        "lnHF": ln_hf,
        "SDNN": sdnn,
        "Artifact %": artifact_ratio * 100
    }, None


# -------------------------
# Effect Size
# -------------------------
def cohens_d(pre, post):
    diff = post - pre
    return np.mean(diff) / np.std(diff, ddof=1)


# -------------------------
# Run Analysis
# -------------------------
if pre_file and post_file:

    pre_rr = np.loadtxt(pre_file)
    post_rr = np.loadtxt(post_file)

    if st.button("Run Pre–Post Analysis"):

        pre_results, pre_error = compute_hrv(pre_rr)
        post_results, post_error = compute_hrv(post_rr)

        if pre_error or post_error:
            st.error("One dataset excluded due to artifact >5%")
        else:

            variables = ["Mean HR", "RMSSD", "lnRMSSD", "HF", "lnHF", "SDNN"]

            rows = []

            for var in variables:
                pre_val = pre_results[var]
                post_val = post_results[var]
                delta = post_val - pre_val

                t_stat, p_val = ttest_rel([pre_val], [post_val])
                d = cohens_d(np.array([pre_val]), np.array([post_val]))

                rows.append([
                    var,
                    round(pre_val, 3),
                    round(post_val, 3),
                    round(delta, 3),
                    round(p_val, 4),
                    round(d, 3)
                ])

            df = pd.DataFrame(rows,
                columns=["Variable", "Pre", "Post", "Δ", "p-value", "Cohen's d"]
            )

            st.success("Analysis Complete")
            st.dataframe(df)
