# hrv_analyzer.py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
from typing import Dict, Tuple, List
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class HRVResult:
    """HRV 분석 결과 저장 클래스"""
    rmssd: float
    sdnn: float
    sdsd: float
    nn50: float
    pnn50: float
    lf: float
    hf: float
    lf_hf_ratio: float
    mean_rr: float
    std_rr: float
    quality_score: float
    artifact_count: int
    processing_time: float


class HRVAnalyzer:
    """HRV 분석 메인 클래스"""
    
    def __init__(self, config: Dict = None):
        """
        Args:
            config (Dict): HRV 분석 설정값
        """
        self.config = config or {}
        self.rr_intervals = None
        self.cleaned_rr = None
        self.quality_score = 0.0
        
    def load_rr_data(self, df: pd.DataFrame, rr_column: str = 'RR') -> bool:
        """
        RR interval 데이터 로드 및 검증
        
        Args:
            df (pd.DataFrame): RR 데이터프레임
            rr_column (str): RR 컬럼 이름
            
        Returns:
            bool: 데이터 유효 여부
        """
        try:
            if rr_column not in df.columns:
                # 가능한 컬럼명 자동 탐지
                possible_cols = [col for col in df.columns 
                               if col.lower() in ['rr', 'rr interval', 'rr_interval', 'rrinterval']]
                if possible_cols:
                    rr_column = possible_cols[0]
                else:
                    raise ValueError(f"RR 컬럼을 찾을 수 없습니다. 가능한 컬럼: {df.columns.tolist()}")
            
            # RR 데이터 추출 및 정제
            self.rr_intervals = df[rr_column].dropna().values.astype(float)
            
            # 유효성 검증
            if len(self.rr_intervals) < self.config.get('min_rr_length', 300):
                raise ValueError(f"RR 데이터가 너무 짧습니다. ({len(self.rr_intervals)} < 300)")
            
            return True
            
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
            return False
    
    def detect_artifacts(self) -> Tuple[np.ndarray, List[int]]:
        """
        Artifact(이상값) 감지
        
        Returns:
            Tuple[np.ndarray, List[int]]: (정제된 RR, artifact 인덱스)
        """
        if self.rr_intervals is None:
            raise ValueError("RR 데이터가 로드되지 않았습니다.")
        
        cleaned_rr = self.rr_intervals.copy()
        artifact_indices = []
        
        # 연속 RR 차이 계산
        rr_diff = np.abs(np.diff(cleaned_rr))
        mean_diff = np.mean(rr_diff)
        std_diff = np.std(rr_diff)
        
        # Threshold 설정
        threshold = mean_diff + 2.5 * std_diff
        
        # Artifact 감지
        anomalies = np.where(rr_diff > threshold)[0]
        
        for idx in anomalies:
            if cleaned_rr[idx] > cleaned_rr[idx + 1]:
                artifact_indices.append(idx)
                cleaned_rr[idx] = np.mean([cleaned_rr[idx - 1], cleaned_rr[idx + 1]]) if idx > 0 else cleaned_rr[idx + 1]
            else:
                artifact_indices.append(idx + 1)
                cleaned_rr[idx + 1] = np.mean([cleaned_rr[idx], cleaned_rr[idx + 2]]) if idx + 2 < len(cleaned_rr) else cleaned_rr[idx]
        
        self.cleaned_rr = cleaned_rr
        self.quality_score = 1.0 - (len(artifact_indices) / len(self.rr_intervals))
        
        return cleaned_rr, artifact_indices
    
    def calculate_time_domain(self) -> Dict[str, float]:
        """
        시간 영역(Time Domain) HRV 지표 계산
        
        Returns:
            Dict: RMSSD, SDNN, SDSD, NN50, pNN50
        """
        if self.cleaned_rr is None:
            self.detect_artifacts()
        
        rr = self.cleaned_rr
        
        # RMSSD
        rr_diff = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))
        
        # SDNN
        sdnn = np.std(rr)
        
        # SDSD
        sdsd = np.std(rr_diff)
        
        # NN50
        nn50 = np.sum(np.abs(rr_diff) > 50)
        
        # pNN50
        pnn50 = (nn50 / (len(rr) - 1)) * 100 if len(rr) > 1 else 0
        
        return {
            "rmssd": float(rmssd),
            "sdnn": float(sdnn),
            "sdsd": float(sdsd),
            "nn50": float(nn50),
            "pnn50": float(pnn50),
            "mean_rr": float(np.mean(rr)),
            "std_rr": float(np.std(rr)),
        }
    
    def calculate_frequency_domain(self) -> Dict[str, float]:
        """
        주파수 영역(Frequency Domain) HRV 지표 계산
        
        Returns:
            Dict: LF, HF, LF/HF ratio
        """
        if self.cleaned_rr is None:
            self.detect_artifacts()
        
        rr = self.cleaned_rr
        
        # RR interval을 일정한 샘플링 레이트로 보간
        sampling_rate = 4.0
        t_orig = np.cumsum(rr) / 1000.0
        t_interp = np.arange(0, t_orig[-1], 1/sampling_rate)
        
        # 선형 보간
        rr_interp = np.interp(t_interp, t_orig, rr)
        
        # Detrending
        rr_detrended = signal.detrend(rr_interp)
        
        # FFT 계산
        fft_values = np.abs(fft(rr_detrended)) ** 2
        fft_freq = fftfreq(len(rr_detrended), 1/sampling_rate)
        
        # 양수 주파수만 사용
        positive_freq = fft_freq[:len(fft_freq)//2]
        positive_power = fft_values[:len(fft_values)//2]
        
        # LF, HF 대역 계산
        lf_band = self.config.get('lf_band', (0.04, 0.15))
        hf_band = self.config.get('hf_band', (0.15, 0.4))
        
        lf_mask = (positive_freq >= lf_band[0]) & (positive_freq < lf_band[1])
        hf_mask = (positive_freq >= hf_band[0]) & (positive_freq < hf_band[1])
        
        lf = np.sum(positive_power[lf_mask])
        hf = np.sum(positive_power[hf_mask])
        
        lf_hf_ratio = lf / hf if hf > 0 else 0.0
        
        return {
            "lf": float(lf),
            "hf": float(hf),
            "lf_hf_ratio": float(lf_hf_ratio),
        }
    
    def analyze(self) -> HRVResult:
        """
        전체 HRV 분석 수행
        
        Returns:
            HRVResult: 완전한 HRV 분석 결과
        """
        import time
        start_time = time.time()
        
        try:
            # Artifact 감지
            cleaned_rr, artifact_count = self.detect_artifacts()
            
            # 시간 영역 분석
            time_domain = self.calculate_time_domain()
            
            # 주파수 영역 분석
            freq_domain = self.calculate_frequency_domain()
            
            # 결과 통합
            result = HRVResult(
                rmssd=time_domain["rmssd"],
                sdnn=time_domain["sdnn"],
                sdsd=time_domain["sdsd"],
                nn50=time_domain["nn50"],
                pnn50=time_domain["pnn50"],
                lf=freq_domain["lf"],
                hf=freq_domain["hf"],
                lf_hf_ratio=freq_domain["lf_hf_ratio"],
                mean_rr=time_domain["mean_rr"],
                std_rr=time_domain["std_rr"],
                quality_score=self.quality_score,
                artifact_count=len(artifact_count),
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            print(f"분석 오류: {str(e)}")
            raise
    
    def get_quality_assessment(self) -> Dict[str, any]:
        """데이터 품질 평가"""
        assessment = {
            "quality_score": self.quality_score,
            "quality_level": "High" if self.quality_score >= 0.95 else 
                           "Moderate" if self.quality_score >= 0.80 else
                           "Low" if self.quality_score >= 0.60 else "Very Low",
            "artifact_percentage": (1 - self.quality_score) * 100,
            "recommendation": self._get_recommendation()
        }
        return assessment
    
    def _get_recommendation(self) -> str:
        """품질 기반 권장사항"""
        if self.quality_score >= 0.95:
            return "✅ 데이터 품질 우수 - 분석 가능"
        elif self.quality_score >= 0.80:
            return "⚠️ 데이터 품질 양호 - 분석 가능"
        elif self.quality_score >= 0.60:
            return "⚠️ 데이터 품질 낮음 - 신중한 해석 필요"
        else:
            return "❌ 데이터 품질 매우 낮음 - 분석 부적합"
