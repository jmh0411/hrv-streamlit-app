# config.py
import os
from pathlib import Path

# 기본 설정
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / ".streamlit_cache"

# HRV 분석 설정 (노인 대상 연구)
HRV_CONFIG = {
    "rr_unit": "ms",
    "detrending": True,
    "artifact_threshold": 0.2,
    "fft_resolution": 0.25,
    "lf_band": (0.04, 0.15),
    "hf_band": (0.15, 0.4),
}

# Streamlit 설정
STREAMLIT_CONFIG = {
    "page_title": "HRV Analysis Tool",
    "page_icon": "❤️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# 데이터 검증 기준
VALIDATION_RULES = {
    "min_rr_length": 300,
    "min_rr_value": 300,
    "max_rr_value": 2000,
    "quality_threshold": 0.8,
}
