import streamlit as st
import pandas as pd
import numpy as np

st.title("HRV Research Data Manager")

# -----------------------------
# 세션 초기화
# -----------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()

# -----------------------------
# ID 입력
# -----------------------------
st.subheader("Participant ID")

participant_id = st.text_input("Enter Participant ID")

# -----------------------------
# HRV 값 입력
# -----------------------------
st.subheader("HRV Metrics")

mean_rr = st.number_input("Mean RR (ms)", value=0.0)
mean_hr = st.number_input("Mean HR (bpm)", value=0.0)
sdnn = st.number_input("SDNN (ms)", value=0.0)
rmssd = st.number_input("RMSSD (ms)", value=0.0)
pnn50 = st.number_input("pNN50 (%)", value=0.0)
lf = st.number_input("LF Power", value=0.0)
hf = st.number_input("HF Power", value=0.0)
lfhf = st.number_input("LF/HF Ratio", value=0.0)

# -----------------------------
# 저장 (중복 방지 + 업데이트 구조)
# -----------------------------
if st.button("Save / Update Participant"):

    if participant_id == "":
        st.warning("ID를 입력하세요.")
    else:

        new_data = {
            "ID": participant_id,
            "Mean RR (ms)": mean_rr,
            "Mean HR (bpm)": mean_hr,
            "SDNN (ms)": sdnn,
            "RMSSD (ms)": rmssd,
            "pNN50 (%)": pnn50,
            "LF Power": lf,
            "HF Power": hf,
            "LF/HF Ratio": lfhf
        }

        df = st.session_state.dataset

        # ID 존재 여부 확인
        if participant_id in df["ID"].values:
            # 업데이트
            df.loc[df["ID"] == participant_id] = new_data
            st.success("기존 데이터 업데이트 완료")
        else:
            # 새로 추가
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            st.session_state.dataset = df
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
st.subheader("Delete Participant")

delete_id = st.text_input("삭제할 ID 입력")

if st.button("Delete"):

    df = st.session_state.dataset

    if delete_id in df["ID"].values:
        df = df[df["ID"] != delete_id]
        st.session_state.dataset = df.reset_index(drop=True)
        st.success("삭제 완료")
    else:
        st.warning("해당 ID 없음")

# -----------------------------
# 그룹 통계 자동 계산
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
st.subheader("Download Full Dataset")

if not st.session_state.dataset.empty:
    csv = st.session_state.dataset.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Full Dataset CSV",
        data=csv,
        file_name="hrv_full_dataset.csv",
        mime="text/csv"
    )
