# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from hrv_analyzer import HRVAnalyzer, HRVResult
from utils import validate_csv_format, detect_rr_column, export_results, create_rr_tachogram_data
from config import STREAMLIT_CONFIG, HRV_CONFIG, VALIDATION_RULES
import time

# Streamlit 페이지 설정
st.set_page_config(**STREAMLIT_CONFIG)

# CSS 스타일
st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 페이지 제목
st.title("❤️ HRV Analysis Tool for Elderly Research")
st.markdown("**RR interval CSV 기반 심박변이도(HRV) 분석**")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("RR interval CSV 파일 업로드", type=["csv"])
    
    st.divider()
    
    # 고급 설정
    with st.expander("🔧 고급 설정"):
        detrending = st.checkbox("Detrending 적용", value=True)
        artifact_threshold = st.slider("Artifact 감지 임계값", 
                                       min_value=1.5, max_value=4.0, 
                                       value=2.5, step=0.1)
        lf_low = st.number_input("LF 대역 최소값 (Hz)", value=0.04, step=0.01)
        lf_high = st.number_input("LF 대역 최대값 (Hz)", value=0.15, step=0.01)
        hf_low = st.number_input("HF 대역 최소값 (Hz)", value=0.15, step=0.01)
        hf_high = st.number_input("HF 대역 최대값 (Hz)", value=0.4, step=0.01)
    
    st.divider()
    
    st.markdown("### 📋 HRV 참고값 (노인)")
    st.info("""
    - **RMSSD**: 정상 > 20ms
    - **SDNN**: 정상 > 50ms
    - **LF/HF Ratio**: 0.5-2.0 (정상범위)
    """)

# 메인 콘텐츠
if uploaded_file is None:
    st.info("👈 좌측 패널에서 RR interval CSV 파일을 업로드하세요.")
    
    # 샘플 데이터 표시
    with st.expander("📚 CSV 파일 형식 안내"):
        st.markdown("""
        ### 필요한 파일 형식:
        
        **Option 1: RR interval (권장)**
        ```
        RR
        850
        820
        900
        ...
        ```
        """)
        
        # 샘플 파일 생성
        sample_data = pd.DataFrame({
            'RR': np.random.normal(800, 50, 300).astype(int)
        })
        sample_csv = sample_data.to_csv(index=False)
        
        st.download_button(
            label="📥 샘플 CSV 다운로드",
            data=sample_csv,
            file_name="sample_rr_data.csv",
            mime="text/csv"
        )

else:
    # 파일 읽기
    try:
        df = pd.read_csv(uploaded_file)
        
        # 포맷 검증
        is_valid, validation_msg = validate_csv_format(df)
        
        if not is_valid:
            st.error(f"❌ {validation_msg}")
        else:
            st.success(f"✅ {validation_msg}")
            
            # RR 컬럼 탐지
            rr_column = detect_rr_column(df)
            
            st.subheader(f"📊 데이터 미리보기")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 RR 개수", len(df[rr_column].dropna()))
            with col2:
                st.metric("데이터 범위", f"{df[rr_column].min():.0f} - {df[rr_column].max():.0f} ms")
            with col3:
                st.metric("평균 RR", f"{df[rr_column].mean():.0f} ms")
            
            st.dataframe(df.head(10))
            
            # 분석 실행 버튼
            if st.button("🚀 HRV 분석 실행", key="analyze_button"):
                
                with st.spinner("분석 중..."):
                    # HRV 분석기 초기화
                    config = {
                        **HRV_CONFIG,
                        'lf_band': (lf_low, lf_high),
                        'hf_band': (hf_low, hf_high),
                    }
                    
                    analyzer = HRVAnalyzer(config=config)
                    
                    # 데이터 로드
                    if not analyzer.load_rr_data(df, rr_column):
                        st.error("데이터 로드 실패")
                    else:
                        # 분석 수행
                        result = analyzer.analyze()
                        quality_assessment = analyzer.get_quality_assessment()
                        
                        st.success("✅ 분석 완료!")
                        
                        # 결과 저장
                        st.session_state.result = result
                        st.session_state.analyzer = analyzer
                        st.session_state.quality = quality_assessment
            
            # 분석 결과 표시
            if 'result' in st.session_state:
                result = st.session_state.result
                analyzer = st.session_state.analyzer
                quality = st.session_state.quality
                
                st.divider()
                st.subheader("📈 분석 결과")
                
                # 품질 평가
                st.markdown("### 📊 데이터 품질")
                
                quality_color = {
                    "High": "🟢",
                    "Moderate": "🟡",
                    "Low": "🔴",
                    "Very Low": "⛔"
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("품질 점수", f"{quality['quality_score']:.2%}")
                with col2:
                    st.metric("품질 레벨", f"{quality_color[quality['quality_level']]} {quality['quality_level']}")
                with col3:
                    st.metric("Artifact 비율", f"{quality['artifact_percentage']:.1f}%")
                
                st.info(quality['recommendation'])
                
                # 시간 영역 지표
                st.markdown("### ⏱️ 시간 영역(Time Domain) 지표")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSSD", f"{result.rmssd:.2f} ms")
                with col2:
                    st.metric("SDNN", f"{result.sdnn:.2f} ms")
                with col3:
                    st.metric("SDSD", f"{result.sdsd:.2f} ms")
                with col4:
                    st.metric("pNN50", f"{result.pnn50:.2f} %")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("NN50", f"{result.nn50:.0f}")
                with col2:
                    st.metric("평균 RR", f"{result.mean_rr:.2f} ms")
                
                # 주파수 영역 지표
                st.markdown("### 📡 주파수 영역(Frequency Domain) 지표")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("LF Power", f"{result.lf:.2e}")
                with col2:
                    st.metric("HF Power", f"{result.hf:.2e}")
                with col3:
                    st.metric("LF/HF Ratio", f"{result.lf_hf_ratio:.2f}")
                
                # 시각화
                st.markdown("### 📊 시각화")
                
                tab1, tab2 = st.tabs(["RR Tachogram", "히스토그램"])
                
                with tab1:
                    rr_data = create_rr_tachogram_data(analyzer.cleaned_rr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=rr_data['Time (s)'],
                        y=rr_data['RR Interval (ms)'],
                        mode='lines+markers',
                        name='RR Interval',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig.update_layout(
                        title="RR Tachogram",
                        xaxis_title="시간 (초)",
                        yaxis_title="RR Interval (ms)",
                        hovermode='x unified',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = px.histogram(
                        rr_data,
                        x='RR Interval (ms)',
                        nbins=30,
                        title="RR Interval 분포"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 다운로드 버튼
                st.markdown("### 💾 결과 다운로드")
                
                csv_data = export_results(result)
                st.download_button(
                    label="📥 결과를 CSV로 다운로드",
                    data=csv_data,
                    file_name="hrv_analysis_results.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.info("파일 형식을 확인해주세요.")

# 페이지 하단
st.divider()
st.markdown("""
---
**⚠️ 의료 면책 조항**
이 도구는 연구 목적의 HRV 분석을 지원합니다.
""")
