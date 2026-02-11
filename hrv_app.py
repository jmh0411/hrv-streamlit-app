# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal, interpolate
import plotly.graph_objects as go

# ------------------------------
# 유틸 함수
# ------------------------------
def infer_units_and_to_ms(rr_series):
    rr = pd.to_numeric(rr_series, errors='coerce').dropna().values.astype(float)
    if len(rr) == 0:
        return rr
    med = np.median(rr)
    if med < 10:  # 초 단위이면 ms로 변환
        rr_ms = rr * 1000.0
    else:
        rr_ms = rr
    return rr_ms

def physiological_mask(rr_ms, rr_min=300, rr_max=2000):
    return (rr_ms >= rr_min) & (rr_ms <= rr_max)

def rolling_median(arr, window=11):
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    med = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        med[i] = np.median(padded[i:i+window])
    return med

def mad_based_z_scores(rr_ms):
    med = np.median(rr_ms)
    mad = np.median(np.abs(rr_ms - med))
    if mad == 0:
        return np.zeros_like(rr_ms)
    z = np.abs(rr_ms - med) / (1.4826 * mad)
    return z

def detect_artifacts(rr_ms, rel_thresh=0.20, abs_thresh_ms=200, window=11, mad_z_thresh=5.0):
    n = len(rr_ms)
    if n < 5:
        return np.zeros(n, dtype=bool)
    phys_ok = physiological_mask(rr_ms)
    med_local = rolling_median(rr_ms, window=window)
    diff = np.abs(rr_ms - med_local)
    local_ok = diff <= np.maximum(rel_thresh * med_local, abs_thresh_ms)
    drr = np.diff(rr_ms, prepend=rr_ms[0])
    global_med = np.median(rr_ms)
    jump_ok = np.abs(drr) <= (0.20 * global_med)
    z = mad_based_z_scores(rr_ms)
    mad_ok = z <= mad_z_thresh
    ok = phys_ok & local_ok & jump_ok & mad_ok
    return ~ok  # True = artifact

def interpolate_nn(rr_ms, art_mask, max_gap_ms=3000):
    t = np.cumsum(rr_ms) / 1000.0  # seconds
    valid = ~art_mask
    if valid.sum() < 3:
        return rr_ms.copy(), t, {"too_many_artifacts": True, "max_gap_s": None}
    t_valid = t[valid]
    rr_valid = rr_ms[valid]
    kind = 'linear' if len(t_valid) < 4 else 'cubic'
    f = interpolate.interp1d(t_valid, rr_valid, kind=kind, bounds_error=False, fill_value="extrapolate")
    rr_nn = rr_ms.copy()
    rr_interp = f(t)
    rr_nn[art_mask] = rr_interp[art_mask]
    max_gap_s = 0.0
    if art_mask.any():
        idx = np.where(art_mask)[0]
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in groups:
            gap_ms = rr_ms[g].sum()
            max_gap_s = max(max_gap_s, gap_ms / 1000.0)
    flags = {
        "too_many_artifacts": False,
        "max_gap_s": max_gap_s
    }
    return rr_nn, t, flags

def time_domain_metrics(rr_nn_ms):
    rr = rr_nn_ms.astype(float)
    sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
    diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else np.nan
    return sdnn, rmssd

def frequency_domain_lfhf(rr_nn_ms, fs_resample=4.0):
    t = np.cumsum(rr_nn_ms) / 1000.0
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_resample)
    kind = 'linear' if len(rr_nn_ms) < 4 else 'cubic'
    f = interpolate.interp1d(t, rr_nn_ms, kind=kind, bounds_error=False, fill_value="extrapolate")
    rr_uniform = f(t_uniform)
    rr_uniform = rr_uniform - np.mean(rr_uniform)
    nperseg = min(256, len(rr_uniform))
    if nperseg < 64:
        return np.nan, np.nan, np.nan, None, None
    freqs, pxx = signal.welch(rr_uniform, fs=fs_resample, window='hann',
                              nperseg=nperseg, noverlap=nperseg // 2,
                              detrend=False, scaling='density')
    # **절대 np.trapz만 사용, scipy.trapz 호출 없음**
    def bandpower(f, p, fmin, fmax):
        band = (f >= fmin) & (f < fmax)
        if not np.any(band):
            return np.nan
        return np.trapz(p[band], f[band])
    lf = bandpower(freqs, pxx, 0.04, 0.15)
    hf = bandpower(freqs, pxx, 0.15, 0.40)
    lfhf = (lf / hf) if (hf is not None and hf > 0) else np.nan
    return lfhf, lf, hf, freqs, pxx

def quality_report(art_mask, flags, total_beats):
    corrected_pct = (art_mask.sum() / total_beats) * 100.0
    notes = []
    if corrected_pct > 10:
        notes.append("아티팩트 보정률 > 10%: 시간영역/주파수영역 지표 신뢰 낮음(제외 권고).")
    elif corrected_pct > 5:
        notes.append("아티팩트 보정률 > 5%: 주파수영역(LF/HF) 해석 주의 또는 제외.")
    if flags.get("max_gap_s", 0) > 3.0:
        notes.append("단일 연속 누락 > 3초: PSD 왜곡 가능성. 해석 주의.")
    return corrected_pct, notes

def parse_uploaded(file) -> pd.DataFrame:
    content = file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), sep='\t')
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'rr' in cols:
        rr_col = 'rr'
    elif 'rr_ms' in cols:
        rr_col = 'rr_ms'
    elif 'ibi' in cols:
        rr_col = 'ibi'
    else:
        if df.shape[1] == 1:
            rr_col = df.columns[0]
        else:
            raise ValueError("RR 컬럼(rr/rr_ms/ibi)을 찾을 수 없습니다.")
    return df[[rr_col]].rename(columns={rr_col: 'rr'})

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="HRV(5분) 분석기", layout="wide")
st.title("5분 HRV 분석(고령자 연구용)")

uploaded = st.file_uploader("RR 파일 업로드(CSV/TXT)", type=["csv","txt"])
run = st.button("분석 실행")

if uploaded and run:
    try:
        df_raw = parse_uploaded(uploaded)
        rr_ms = infer_units_and_to_ms(df_raw['rr'])
        n_beats = len(rr_ms)
        st.write(f"총 비트 수: {n_beats}")

        # artifact detection & interpolation
        art_mask = detect_artifacts(rr_ms)
        rr_nn, t_s, flags = interpolate_nn(rr_ms, art_mask)

        # metrics
        sdnn, rmssd = time_domain_metrics(rr_nn)
        lfhf, lf, hf, freqs, pxx = frequency_domain_lfhf(rr_nn)

        corrected_pct, notes = quality_report(art_mask, flags, n_beats)
        st.subheader("품질 지표")
        st.write(f"아티팩트 보정률: {corrected_pct:.2f}%")
        st.write(f"최장 연속 누락: {flags['max_gap_s']:.2f} s")
        for n in notes:
            st.warning(n)

        st.subheader("HRV 지표")
        st.write(f"SDNN: {sdnn:.2f} ms, RMSSD: {rmssd:.2f} ms, LF/HF: {lfhf:.2f}")

    except Exception as e:
        st.error(f"분석 중 오류: {str(e)}")
