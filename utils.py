# utils.py
import pandas as pd
import numpy as np
from typing import Tuple, List
import io

def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    CSV 파일 포맷 검증
    
    Returns:
        Tuple[bool, str]: (유효성, 메시지)
    """
    if df.empty:
        return False, "CSV 파일이 비어있습니다."
    
    # RR 컬럼 탐지
    possible_columns = ['RR', 'RR interval', 'rr_interval', 'rrinterval', 
                       'IBI', 'nn interval', 'Beat-to-beat']
    
    rr_column = None
    for col in df.columns:
        if col.lower() in [c.lower() for c in possible_columns]:
            rr_column = col
            break
    
    if rr_column is None:
        return False, f"RR interval 컬럼을 찾을 수 없습니다. 제공된 컬럼: {', '.join(df.columns)}"
    
    # 숫자형 데이터 검증
    try:
        df[rr_column] = pd.to_numeric(df[rr_column], errors='coerce')
        if df[rr_column].isna().sum() > len(df) * 0.5:
            return False, "50% 이상의 데이터가 숫자 형식이 아닙니다."
    except:
        return False, "RR interval을 숫자로 변환할 수 없습니다."
    
    return True, "유효한 형식입니다."


def detect_rr_column(df: pd.DataFrame) -> str:
    """자동으로 RR 컬럼명 탐지"""
    possible_names = ['RR', 'RR interval', 'rr_interval', 'rrinterval', 
                     'IBI', 'nn', 'NN', 'nn_interval']
    
    for col in df.columns:
        if col in possible_names or col.lower() in [n.lower() for n in possible_names]:
            return col
    
    return df.columns[0] if len(df.columns) > 0 else None


def export_results(result, filename: str = "hrv_analysis_results.csv") -> bytes:
    """
    분석 결과를 CSV로 내보내기
    
    Args:
        result: HRVResult 객체
        filename: 파일명
        
    Returns:
        bytes: CSV 파일 바이트
    """
    data = {
        'Metric': ['RMSSD (ms)', 'SDNN (ms)', 'SDSD (ms)', 'NN50', 'pNN50 (%)',
                  'Mean RR (ms)', 'Std RR (ms)', 'LF Power', 'HF Power', 'LF/HF Ratio',
                  'Quality Score', 'Artifact Count'],
        'Value': [result.rmssd, result.sdnn, result.sdsd, result.nn50, result.pnn50,
                 result.mean_rr, result.std_rr, result.lf, result.hf, result.lf_hf_ratio,
                 result.quality_score, result.artifact_count]
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode()


def create_rr_tachogram_data(rr_intervals: np.ndarray) -> pd.DataFrame:
    """RR tachogram 시각화용 데이터 생성"""
    t = np.cumsum(rr_intervals) / 1000.0  # 초 단위
    df = pd.DataFrame({
        'Time (s)': t,
        'RR Interval (ms)': rr_intervals,
        'Heart Rate (bpm)': 60000 / rr_intervals
    })
    return df
