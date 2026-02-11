import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid

st.title("HRV Research Analysis System")

# -----------------------------
# 세션 초기화
# -----------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()

# -----------------------------
# RR 업로드
# -----------------------------
st.subheader("Upload RR Interval CSV (ms)")

uploaded_file = st.file_uploader("Upload RR data", type=["csv"])

participant_id = st.text_input("Participant ID")

# -----------------------------
# HRV 계산 함수
# -----------------------------
def calculate_hrv(rr):

    rr = np.array(rr, dtype=float)

    # Artifact 제거 (Kubios guideline 기반 physiological filter)
    rr = rr[(rr > 300) & (rr < 2000)]

    if len(rr) < 60:
        return None

    mean_rr = np.mean(rr)
    mean_hr = 60000 / mean_rr
    sdnn = np.std(rr, ddof=1)

    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100

    # --- Frequency domain (Welch)
    time_axis = np.cumsum(rr) / 1000
    fs = 4
    interpolated_time = np.arange(0, time_axis[-1], 1/fs)
    interpolated_rr = np.interp(interpolated_time, time_axis, rr)

    f, pxx = welch(interpolated_rr, fs=fs, nperseg=256)

    lf_band = (f >= 0.04) & (f <= 0.15)
    hf_band = (f >= 0.15) & (f <= 0.4)

    lf = trapezoid(pxx[lf_band], f[lf_band])
    hf = trapezoid(pxx[hf_band], f[hf_band])

    lfhf = lf / hf if hf != 0 else np.nan

    return {
        "Mean RR (ms)": round(mean_rr,2),
        "Mean HR (bpm)": round(mean_hr,2),
        "SDNN (ms)": round(sdnn,2),
        "RMSSD (ms)": round(rmssd,2),
        "pNN50 (%)": round(pnn50,2),
        "LF Power": round(lf,2),
        "HF Power": round(hf,2),
        "LF/HF Ratio": round(lfhf,2)
    }

# -----------------------------
# 분석 실행
# -----------------------------
if uploaded_file and participant_id:

    df = pd.read_csv(uploaded_file)

    rr_column = df.columns[0]
    rr_data = df[rr_column].dropna()

    results = calculate_hrv(rr_data)

    if results is None:
        st.error("RR 데이터 길이 부족 또는 품질 문제")
    else:
        st.subheader("Calculated HRV Results")
        st.json(results)

        if st.button("Save / Update Participant"):

            results["ID"] = participant_id
            new_df = pd.DataFrame([results])

            dataset = st.session_state.dataset

            if not dataset.empty and participant_id in dataset["ID"].values:
                dataset.loc[dataset["ID"] == participant_id] = new_df.values
                st.success("기존 데이터 업데이트 완료")
            else:
                dataset = pd.concat([dataset, new_df], ignore_index=True)
                st.session_state.dataset = dataset
                st.success("새 데이터 저장 완료")

# -----------------------------
# 현재 데이터 표시
# -----------------------------
st.subheader("Current Dataset")

if not st.session_state.dataset.empty:
    st.dataframe(st.session_state.dataset)

# -----------------------------
# 삭제 기능
# -----------------------------
delete_id = st.text_input("삭제할 ID 입력")

if st.button("Delete Participant"):

    df = st.session_state.dataset

    if delete_id in df["ID"].values:
        df = df[df["ID"] != delete_id]
        st.session_state.dataset = df.reset_index(drop=True)
        st.success("삭제 완료")
    else:
        st.warning("해당 ID 없음")

# -----------------------------
# 그룹 통계
# -----------------------------
st.subheader("Group Statistics")

if not st.session_state.dataset.empty:

    numeric_df = st.session_state.dataset.drop(columns=["ID"])

    stats = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "SD": numeric_df.std(),
        "Min": numeric_df.min(),
        "Max": numeric_df.max()
    })

    st.dataframe(stats)

# -----------------------------
# CSV 다운로드
# -----------------------------
if not st.session_state.dataset.empty:
    csv = st.session_state.dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Dataset CSV",
        data=csv,
        file_name="hrv_full_dataset.csv",
        mime="text/csv"
    )
