import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

st.title("Single Upload HRV Analysis (Pre–Post)")

remove_first_min = st.checkbox("Remove first 60 seconds (recommended)", value=True)

uploaded_file = st.file_uploader("Upload RR interval file (.txt)", type=["txt"])

if "pre_data" not in st.session_state:
    st.session_state.pre_data = None
if "post_data" not in st.session_state:
    st.session_state.post_data = None


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

    if remove_first_min:
        t = np.cumsum(rr_corrected) / 1000
        rr_corrected = rr_corrected[t > 60]

    mean_rr = np.mean(rr_corrected)
    mean_hr = 60000 / mean_rr

    sdnn = np.std(rr_corrected, ddof=1)
    diff_rr = np.diff(rr_corrected)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    ln_rmssd = np.log(rmssd)

    # Frequency domain (HF only)
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
# Upload Handling
# -------------------------
if uploaded_file is not None:

    rr = np.loadtxt(uploaded_file)

    results, error = compute_hrv(rr)

    if error:
        st.error(error)
    else:
        st.success("Analysis Complete")
        st.write(results)

        col1, col2 = st.columns(2)

        if col1.button("Save as PRE"):
            st.session_state.pre_data = results
            st.success("Saved as PRE")

        if col2.button("Save as POST"):
            st.session_state.post_data = results
            st.success("Saved as POST")


# -------------------------
# Comparison
# -------------------------
if st.session_state.pre_data and st.session_state.post_data:

    st.subheader("Pre–Post Comparison")

    variables = ["Mean HR", "RMSSD", "lnRMSSD", "HF", "lnHF", "SDNN"]

    rows = []

    for var in variables:
        pre_val = st.session_state.pre_data[var]
        post_val = st.session_state.post_data[var]
        delta = post_val - pre_val

        rows.append([
            var,
            round(pre_val, 3),
            round(post_val, 3),
            round(delta, 3)
        ])

    df = pd.DataFrame(rows,
        columns=["Variable", "Pre", "Post", "Δ"]
    )

    st.dataframe(df)
