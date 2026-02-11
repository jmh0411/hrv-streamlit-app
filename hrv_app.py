# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal, interpolate

# ------------------------------
# 유틸 함수
# ------------------------------
def infer_units_and_to_ms(rr_series):
    rr = pd.to_numeric(rr_series, errors='coerce').dropna().values.astype(float)
    if len(rr) == 0:
        return rr
    med = np.median(rr)
    # 간단한 단위 추정: 초 단위면 중앙값이 보통 < 3 (e.g., 0.8s), ms면 > 100
    if med < 10:
        rr_ms = rr * 1000.0
    else:
        rr_ms = rr
    return rr_ms

def physiological_mask(rr_ms, rr_min=300, rr_max=2000):
    return (rr_ms >= rr_min) & (rr_ms <= rr_max)

def rolling_median(arr, window=11):
    # 중앙값 롤링 (패딩: 가장자리 보완)
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

    # A) 생리적 범위
    phys_ok = physiological_mask(rr_ms)

    # B) 로컬 중앙값 기반 상대편차
    med_local = rolling_median(rr_ms, window=window)
    diff = np.abs(rr_ms - med_local)
    local_ok = diff <= np.maximum(rel_thresh * med_local, abs_thresh_ms)

    # C) ΔRR 급변 필터(보조)
    drr = np.diff(rr_ms, prepend=rr_ms[0])
    global_med = np.median(rr_ms)
    jump_ok = np.abs(drr) <= (0.20 * global_med)

    # D) MAD 기반 필터(보조)
    z = mad_based_z_scores(rr_ms)
    mad_ok = z <= mad_z_thresh

    # 통합(안전하게 교집합 기준으로 OK 결정)
    ok = phys_ok & local_ok & jump_ok & mad_ok

    # 양끝부 등급 보수: 만약 너무 빡빡하면 보정
    return ~ok  # True = artifact

def interpolate_nn(rr_ms, art_mask, max_gap_ms=3000):
    """
    아티팩트 True 지점을 시간축 기반으로 선형 보간.
    길게 이어진 누락(max_gap_ms 초과)은 경고 플래그로 반환.
    """
    t = np.cumsum(rr_ms) / 1000.0  # seconds
    valid = ~art_mask

    if valid.sum() < 3:
        return rr_ms.copy(), t, {"too_many_artifacts": True, "max_gap_s": None}

    # 유효 포인트로 시간-값 보간 함수 생성
    t_valid = t[valid]
    rr_valid = rr_ms[valid]
    kind = 'linear' if len(t_valid) < 4 else 'cubic'
    f = interpolate.interp1d(t_valid, rr_valid, kind=kind, bounds_error=False)

    rr_nn = rr_ms.copy()
    # 경계에서 보간: 범위 밖은 최근접값으로 대체
    left_val = rr_valid[0]
    right_val = rr_valid[-1]
    rr_interp = f(t)
    rr_interp[:np.searchsorted(t, t_valid[0])] = left_val
    rr_interp[np.searchsorted(t, t_valid[-1], side='right'):] = right_val

    rr_nn[art_mask] = rr_interp[art_mask]

    # 최장 연속 누락 길이 계산
    max_gap_s = 0.0
    if art_mask.any():
        # 연속 True 구간 길이(시간)
        idx = np.where(art_mask)[0]
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in groups:
            # 해당 구간의 시간 길이 = 해당 RR들의 합
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
    # 시간축
    t = np.cumsum(rr_nn_ms) / 1000.0  # s
    # 4 Hz 균등 샘플 시간축
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_resample)
    # RR(t) 보간 (cubic 가능)
    kind = 'linear' if len(rr_nn_ms) < 4 else 'cubic'
    f = interpolate.interp1d(t, rr_nn_ms, kind=kind, bounds_error=False, fill_value="extrapolate")
    rr_uniform = f(t_uniform)

    # 평균 제거(기본 detrend)
    rr_uniform = rr_uniform - np.mean(rr_uniform)

    # Welch PSD
    nperseg = min(256, len(rr_uniform))
    if nperseg < 64:
        return np.nan, np.nan, np.nan, None, None
    freqs, pxx = signal.welch(rr_uniform, fs=fs_resample, window='hann',
                              nperseg=nperseg, noverlap=nperseg // 2,
                              detrend=False, scaling='density')

    def bandpower(f, p, fmin, fmax):
        band = (f >= fmin) & (f < fmax)
        if not np.any(band):
            return np.nan
        return np.trapz(p[band], f[band])

    lf = bandpower(freqs, pxx, 0.04, 0.15)
    hf = bandpower(freqs, pxx, 0.15, 0.40)
    lfhf = (lf / hf) if (hf is not None and isinstance(hf, float) and hf > 0) else (lf / hf if (hf and hf > 0) else np.nan)
    return lfhf, lf, hf, freqs, pxx

def quality_report(art_mask, flags, total_beats):
    corrected_pct = (art_mask.sum() / total_beats) * 100.0
    notes = []
    if corrected_pct > 10:
        notes.append("아티팩트 보정률 > 10%: 시간영역/주파수영역 지표 신뢰 낮음(제외 권고).")
    elif corrected_pct > 5:
        notes.append("아티팩트 보정률 > 5%: 주파수영역(LF/HF) 해석 주의 또는 제외.")
    if flags.get("max_gap_s", 0) and flags["max_gap_s"] > 3.0:
        notes.append("단일 연속 누락 > 3초: PSD 왜곡 가능성. 해석 주의.")
    return corrected_pct, notes

def parse_uploaded(file) -> pd.DataFrame:
    # CSV 또는 텍스트 자동 판독
    content = file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), sep='\t')
    # 컬럼 추론
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'rr' in cols:
        rr_col = 'rr'
    elif 'rr_ms' in cols:
        rr_col = 'rr_ms'
    elif 'ibi' in cols:  # inter-beat interval
        rr_col = 'ibi'
    else:
        # 단일 컬럼인 경우 가정
        if df.shape[1] == 1:
            rr_col = df.columns[0]
        else:
            raise ValueError("RR 데이터를 담은 컬럼(rr, rr_ms, ibi 등)을 찾을 수 없습니다.")
    return df[[rr_col]].rename(columns={rr_col: 'rr'})

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="HRV(5분) 분석기", layout="wide")
st.title("5분 HRV 분석(고령자 연구용)")

with st.expander("설정 옵션", expanded=False):
    rr_min = st.number_input("생리적 RR 최소(ms)", value=300, step=50)
    rr_max = st.number_input("생리적 RR 최대(ms)", value=2000, step=50)
    rel_thresh = st.slider("로컬 중앙값 상대 임계(%)", min_value=5, max_value=50, value=20) / 100.0
    abs_thresh_ms = st.number_input("로컬 절대 임계(ms)", value=200, step=10)
    window = st.number_input("로컬 중앙값 윈도우(비트)", value=11, step=2)
    mad_z_thresh = st.number_input("MAD z-score 임계", value=5.0, step=0.5)
    max_gap_ms = st.number_input("허용 단일 연속 누락 최대(ms)", value=3000, step=500)
    fs_resample = st.selectbox("PSD 재표본화 주파수(Hz)", options=[2.0, 4.0], index=1)

uploaded = st.file_uploader("RR 파일 업로드(CSV/TXT). 컬럼명 rr/rr_ms/ibi 또는 단일컬럼", type=["csv", "txt"])
run = st.button("분석 실행")

if uploaded and run:
    try:
        df_raw = parse_uploaded(uploaded)
        rr_ms = infer_units_and_to_ms(df_raw['rr'])
        n_beats = len(rr_ms)
        st.write(f"총 비트 수: {n_beats}")

        # 길이/시간 체크
        total_time_s = rr_ms.sum() / 1000.0
        st.write(f"측정 길이(추정): {total_time_s:.1f} 초")
        if total_time_s < 240 or total_time_s > 360:
            st.warning("권장 5분(240–360초) 범위를 벗어났습니다. 지표 신뢰가 낮을 수 있습니다.")

        # 아티팩트 탐지
        art_mask = detect_artifacts(rr_ms, rel_thresh=rel_thresh, abs_thresh_ms=abs_thresh_ms,
                                    window=window, mad_z_thresh=mad_z_thresh)

        # 보정
        rr_nn, t_s, flags = interpolate_nn(rr_ms, art_mask, max_gap_ms=max_gap_ms)

        # 품질 리포트
        corrected_pct, notes = quality_report(art_mask, flags, n_beats)
        st.subheader("품질지표")
        st.write(f"아티팩트 보정률: {corrected_pct:.2f}%")
        st.write(f"최장 연속 누락: {flags.get('max_gap_s', 0):.2f} s")
        for n in notes:
            st.error(n)

        # 시간영역
        sdnn, rmssd = time_domain_metrics(rr_nn)

        # 주파수영역
        lfhf, lf, hf, freqs, pxx = frequency_domain_lfhf(rr_nn, fs_resample=fs_resample)

        # 결과 표시
        st.subheader("지표 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("SDNN (ms)", f"{sdnn:.2f}" if not np.isnan(sdnn) else "NaN")
        c2.metric("RMSSD (ms)", f"{rmssd:.2f}" if not np.isnan(rmssd) else "NaN")
        c3.metric("LF/HF", f"{lfhf:.2f}" if (lfhf is not None and not np.isnan(lfhf)) else "NaN")
        st.write(f"LF 파워(0.04–0.15 Hz): {lf:.2f} (상대 단위), HF 파워(0.15–0.40 Hz): {hf:.2f} (상대 단위)")

        # 타코그램 플롯
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(n_beats), y=rr_ms, mode='lines+markers',
                                 name='원시 RR(ms)', line=dict(color='lightgray')))
        if art_mask.any():
            fig.add_trace(go.Scatter(x=np.where(art_mask)[0], y=rr_ms[art_mask], mode='markers',
                                     name='아티팩트', marker=dict(color='red', size=6)))
        fig.add_trace(go.Scatter(x=np.arange(n_beats), y=rr_nn, mode='lines',
                                 name='보정 RR(NN)(ms)', line=dict(color='blue')))
        fig.update_layout(title="RR 타코그램(원시/아티팩트/보정)", xaxis_title="Beat Index", yaxis_title="RR (ms)")
        st.plotly_chart(fig, use_container_width=True)

        # PSD 플롯
        if freqs is not None and pxx is not None:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=freqs, y=pxx, mode='lines', name='PSD'))
            # 대역 음영
            fig2.add_vrect(x0=0.04, x1=0.15, fillcolor="orange", opacity=0.2, line_width=0, annotation_text="LF")
            fig2.add_vrect(x0=0.15, x1=0.40, fillcolor="green", opacity=0.2, line_width=0, annotation_text="HF")
            fig2.update_layout(title="Welch PSD (4 Hz 재표본화)", xaxis_title="Frequency (Hz)", yaxis_title="Power Density")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("PSD 계산에 충분한 포인트가 없습니다.")

        # CSV 내보내기
        out = pd.DataFrame({
            "sdnn_ms": [sdnn],
            "rmssd_ms": [rmssd],
            "lf_power": [lf],
            "hf_power": [hf],
            "lf_hf_ratio": [lfhf],
            "corrected_pct": [corrected_pct],
            "max_gap_s": [flags.get("max_gap_s", np.nan)],
            "total_time_s": [total_time_s],
            "n_beats": [n_beats]
        })
        st.download_button("결과 CSV 다운로드", data=out.to_csv(index=False).encode('utf-8-sig'),
                           file_name="hrv_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"분석 중 오류: {str(e)}")
