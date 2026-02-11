import streamlit as st
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import welch

# ===============================
# 안전한 RR 전처리
# ===============================
def preprocess_rr(rr_series):
    rr = pd.to_numeric(rr_series, errors="coerce")

    total_n = len(rr)

    # 1) 생리적 범위
    rr[(rr < 300) | (rr > 2000)] = np.nan

    # 2) 인접 변화율 20% 초과
    diff_ratio = rr.diff().abs() / rr.shift(1)
    rr[diff_ratio > 0.20] = np.nan

    removed_n = rr.isna().sum()
    removed_ratio = removed_n / total_n if total_n > 0 else 0

    # 선형 보간
    rr_interp = rr.interpolate(method="linear", limit_direction="both")

    valid_n = rr_interp.notna().sum()

    return rr_interp.values, total_n, removed_n, removed_ratio, valid_n


# ===============================
# 시간영역
# ===============================
def time_domain(rr_ms):
    if len(rr_ms) < 2:
        return np.nan, np.nan
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    return sdnn, rmssd


# ===============================
# 주파수영역 (안전 버전)
# ===============================
def freq_domain(rr_ms):
    if len(rr_ms) < 240:
        return np.nan

    rr_sec = rr_ms / 1000.0
    t = np.cumsum(rr_sec)
    t -= t[0]

    if t[-1] <= 0:
        return np.nan

    fs = 4.0
    try:
        interp_func = interpolate.interp1d(
            t, rr_sec, kind="linear", fill_value="extrapolate"
        )
        t_interp = np.arange(0, t[-1], 1/fs)
        rr_interp = interp_func(t_interp)

        f, pxx = welch(rr_interp, fs=fs, nperseg=min(256, len(rr_interp)))

        lf = np.trapz(pxx[(f >= 0.04) & (f < 0.15)],
                      f[(f >= 0.04) & (f < 0.15)])
        hf = np.trapz(pxx[(f >= 0.15) & (f < 0.40)],
                      f[(f >= 0.15) & (f < 0.40)])

        if hf == 0:
            return np.nan

        return lf / hf

    except:
        return np.nan


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="5분 HRV 분석", layout="centered")

st.title("📊 5분 HRV 분석 (단일 RR 파일)")

st.markdown("""
**파일 조건**
- .txt 파일
- 단위: ms
- 한 줄에 RR 값 1개
""")

uploaded_file = st.file_uploader("RR 데이터 파일 업로드", type=["txt"])

if uploaded_file is not None:
    try:
        rr_df = pd.read_csv(uploaded_file, header=None)
        rr_series = rr_df.iloc[:, 0]

        rr_clean, total_n, removed_n, removed_ratio, valid_n = preprocess_rr(rr_series)

        st.subheader("📌 데이터 품질 요약")
        st.write(f"총 RR 개수: {total_n}")
        st.write(f"제거/보간 RR 개수: {removed_n}")
        st.write(f"제거 비율: {removed_ratio*100:.2f}%")
        st.write(f"유효 RR 개수: {valid_n}")

        sdnn, rmssd = time_domain(rr_clean)
        lf_hf = freq_domain(rr_clean)

        st.subheader("✅ HRV 계산 결과")

        if removed_ratio <= 0.05:
            st.success("✔ 논문 분석 기준 통과 (≤5%)")
        else:
            st.warning("⚠ 5% 초과 — 논문용 분석은 권장되지 않음 (참고용 결과)")

        st.metric("SDNN (ms)", f"{sdnn:.2f}" if not np.isnan(sdnn) else "계산 불가")
        st.metric("RMSSD (ms)", f"{rmssd:.2f}" if not np.isnan(rmssd) else "계산 불가")
        st.metric("LF/HF", f"{lf_hf:.2f}" if not np.isnan(lf_hf) else "계산 불가")

    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {e}")
