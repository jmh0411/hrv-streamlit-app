import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

st.title("5분 HRV 분석 (RR 파일 업로드)")

st.write("업로드 조건:")
st.write("- RR interval 텍스트 파일 (.txt)")
st.write("- 단위: ms")
st.write("- 한 줄에 RR 값 1개")

uploaded_file = st.file_uploader("RR 데이터 파일 업로드", type=["txt"])

if uploaded_file is not None:

    try:
        # -------------------------
        # 1️⃣ 파일 읽기
        # -------------------------
        rr = pd.read_csv(uploaded_file, header=None)
        rr = rr.iloc[:, 0].astype(float).values

        total_beats = len(rr)

        # -------------------------
        # 2️⃣ 결측 제거
        # -------------------------
        rr = rr[~np.isnan(rr)]

        # -------------------------
        # 3️⃣ 생리학적 범위 필터 (300~2000ms)
        # -------------------------
        valid_mask = (rr >= 300) & (rr <= 2000)
        valid_rr = rr[valid_mask]

        artifact_ratio = 1 - (len(valid_rr) / total_beats)

        st.write(f"총 박동 수: {total_beats}")
        st.write(f"이상치 비율: {artifact_ratio*100:.2f}%")

        if artifact_ratio > 0.05:
            st.error("결측/이상치 비율이 5% 초과하여 분석 제외")
            st.stop()

        rr = valid_rr

        # -------------------------
        # 4️⃣ Time Domain 계산
        # -------------------------
        sdnn = np.std(rr, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))

        # -------------------------
        # 5️⃣ Frequency Domain 계산
        # -------------------------
        rr_sec = rr / 1000.0
        time = np.cumsum(rr_sec)

        fs = 4  # 4Hz 보간
        time_interp = np.arange(0, time[-1], 1/fs)
        rr_interp = np.interp(time_interp, time, rr_sec)

        freq, psd = welch(rr_interp, fs=fs, nperseg=min(256, len(rr_interp)))

        lf_band = (freq >= 0.04) & (freq < 0.15)
        hf_band = (freq >= 0.15) & (freq < 0.4)

        lf = trapezoid(psd[lf_band], freq[lf_band])
        hf = trapezoid(psd[hf_band], freq[hf_band])

        lf_hf_ratio = lf / hf if hf > 0 else np.nan

        # -------------------------
        # 6️⃣ 결과 출력
        # -------------------------
        st.subheader("Time Domain")
        st.write(f"SDNN: {sdnn:.2f} ms")
        st.write(f"RMSSD: {rmssd:.2f} ms")

        st.subheader("Frequency Domain")
        st.write(f"LF Power: {lf:.4f}")
        st.write(f"HF Power: {hf:.4f}")
        st.write(f"LF/HF Ratio: {lf_hf_ratio:.2f}")

        # -------------------------
        # 7️⃣ PSD 그래프
        # -------------------------
        fig, ax = plt.subplots()
        ax.plot(freq, psd)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_title("Power Spectral Density")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
