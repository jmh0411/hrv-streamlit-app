import numpy as np
import streamlit as st

st.write("Numpy version:", np.__version__)
st.write("Has trapz:", hasattr(np, "trapz"))

import streamlit as st
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import welch

# ===============================
# RR 데이터 전처리 (결측치·이상치 제거)
# ===============================
def clean_rr(rr_series):
    rr = rr_series.copy()

    # 1. 생리적 범위 (Task Force, 1996)
    rr[(rr < 300) | (rr > 2000)] = np.nan

    # 2. 인접 RR 변화율 20% 초과 제거 (Kubios 기준)
    diff_ratio = rr.diff().abs() / rr.shift(1)
    rr[diff_ratio > 0.20] = np.nan

    # 결측 비율 확인
    missing_ratio = rr.isna().mean()
    if missing_ratio > 0.05:
        raise ValueError("결측/이상치 비율이 5%를 초과하여 분석 제외")

    # 3. 선형 보간
    rr_interp = rr.interpolate(method="linear")

    if rr_interp.isna().any():
        raise ValueError("보간 후에도 결측치 존재")

    return rr_interp.values


# ===============================
# 시간 영역 지표
# ===============================
def time_domain(rr_ms):
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    return sdnn, rmssd


# ===============================
# 주파수 영역 지표 (LF/HF)
# ===============================
def freq_domain(rr_ms):
    rr_sec = rr_ms / 1000.0
    t = np.cumsum(rr_sec)
    t -= t[0]

    fs = 4.0  # interpolation frequency (Hz)
    interp_func = interpolate.interp1d(t, rr_sec, kind="cubic")
    t_interp = np.arange(0, t[-1], 1/fs)
    rr_interp = interp_func(t_interp)

    f, pxx = welch(rr_interp, fs=fs, nperseg=256)

    lf = np.trapz(pxx[(f >= 0.04) & (f < 0.15)],
                  f[(f >= 0.04) & (f < 0.15)])
    hf = np.trapz(pxx[(f >= 0.15) & (f < 0.40)],
                  f[(f >= 0.15) & (f < 0.40)])

    lf_hf = lf / hf if hf > 0 else np.nan
    return lf_hf


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="5분 HRV 분석", layout="centered")

st.title("📊 5분 HRV 분석 (RR 파일 업로드)")
st.markdown("""
**업로드 조건**
- RR interval 텍스트 파일 (.txt)
- 단위: ms
- 한 줄에 RR 값 1개
""")

uploaded_file = st.file_uploader(
    "RR 데이터 파일 업로드",
    type=["txt"]
)

if uploaded_file is not None:
    try:
        rr_df = pd.read_csv(uploaded_file, header=None, names=["RR"])
        rr_df["RR"] = pd.to_numeric(rr_df["RR"], errors="coerce")

        st.subheader("📌 원본 RR 데이터 미리보기")
        st.dataframe(rr_df.head())

        rr_clean = clean_rr(rr_df["RR"])

        if len(rr_clean) < 240:
            st.error("유효 RR 수가 240 미만 → 분석 제외")
        else:
            sdnn, rmssd = time_domain(rr_clean)
            lf_hf = freq_domain(rr_clean)

            st.subheader("✅ HRV 분석 결과")
            st.metric("SDNN (ms)", f"{sdnn:.2f}")
            st.metric("RMSSD (ms)", f"{rmssd:.2f}")
            st.metric("LF/HF Ratio", f"{lf_hf:.2f}")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")

