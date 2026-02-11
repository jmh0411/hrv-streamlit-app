import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.stats import ttest_rel

st.title("Group HRV Pre–Post Analysis (Research Grade)")

# -----------------------------
# Session State Initialization
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

remove_first_min = st.checkbox("Remove first 60 seconds", value=True)

participant_id = st.text_input("Participant ID (e.g., P01)")

condition = st.selectbox("Condition", ["PRE", "POST"])

uploaded_file = st.file_uploader("Upload RR interval (.txt)", type=["txt"])


# -----------------------------
# Artifact Correction
# -----------------------------
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


# -----------------------------
# HRV Calculation
# -----------------------------
def compute_hrv(rr):

    rr_corrected, artifact_ratio = artifact_correction(rr)

    if artifact_ratio > 0.05:
        return None

    if remove_first_min:
        t = np.cumsum(rr_corrected) / 1000
        rr_corrected = rr_corrected[t > 60]

    mean_rr = np.mean(rr_corrected)
    mean_hr = 60000 / mean_rr

    sdnn = np.std(rr_corrected, ddof=1)
    diff_rr = np.diff(rr_corrected)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    ln_rmssd = np.log(rmssd)

    # Frequency (HF only)
    t = np.cumsum(rr_corrected) / 1000
    t -= t[0]

    fs = 4
    t_interp = np.arange(0, t[-1], 1/fs)
    cs = CubicSpline(t, rr_corrected)
    rr_interp = cs(t_interp)

    rr_interp -= np.polyval(np.polyfit(t_interp, rr_interp, 1), t_interp)

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
        "Mean_HR": mean_hr,
        "RMSSD": rmssd,
        "lnRMSSD": ln_rmssd,
        "HF": hf_power,
        "lnHF": ln_hf,
        "SDNN": sdnn,
        "Artifact_%": artifact_ratio * 100
    }


# -----------------------------
# Upload & Save
# -----------------------------
if uploaded_file and participant_id:

    rr = np.loadtxt(uploaded_file)
    results = compute_hrv(rr)

    if results is None:
        st.error("Artifact >5% → excluded")
    else:
        results["ID"] = participant_id
        results["Condition"] = condition

        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([results])],
            ignore_index=True
        )

        st.success("Data Saved")


# -----------------------------
# Display Data
# -----------------------------
if not st.session_state.data.empty:
    st.subheader("Accumulated Dataset")
    st.dataframe(st.session_state.data)

    csv = st.session_state.data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Full Dataset CSV",
        csv,
        "hrv_group_dataset.csv",
        "text/csv"
    )


# -----------------------------
# Group Statistics
# -----------------------------
if not st.session_state.data.empty:

    df = st.session_state.data

    if set(["PRE", "POST"]).issubset(df["Condition"].unique()):

        st.subheader("Group Pre–Post Statistics")

        pre = df[df["Condition"] == "PRE"].set_index("ID")
        post = df[df["Condition"] == "POST"].set_index("ID")

        common_ids = pre.index.intersection(post.index)

        if len(common_ids) > 1:

            results_table = []

            variables = ["Mean_HR", "RMSSD", "lnRMSSD", "HF", "lnHF", "SDNN"]

            for var in variables:
                pre_vals = pre.loc[common_ids, var]
                post_vals = post.loc[common_ids, var]

                delta = post_vals - pre_vals
                t_stat, p_val = ttest_rel(pre_vals, post_vals)

                d = delta.mean() / delta.std(ddof=1)

                results_table.append([
                    var,
                    round(pre_vals.mean(), 3),
                    round(post_vals.mean(), 3),
                    round(delta.mean(), 3),
                    round(p_val, 4),
                    round(d, 3)
                ])

            stats_df = pd.DataFrame(
                results_table,
                columns=["Variable", "Pre Mean", "Post Mean", "Δ Mean", "p-value", "Cohen's d"]
            )

            st.dataframe(stats_df)
        else:
            st.info("Need at least 2 matched participants for group statistics.")
