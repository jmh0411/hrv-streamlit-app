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
    return rr * 1000.0 if med < 10 else rr

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
    return np.abs(rr_ms - med) / (1.4826 * mad)

def detect_artifacts(rr_ms, rel_thresh=0.20, abs_thresh_ms=200, window=11, mad_z_thresh=5.0):
    n = len(rr_ms)
    if n < 5:
        return np.zeros(n, dtype=bool)
    phys_ok = physiological_mask(rr_ms)
    med_local = rolling_median(rr_ms, window)
    local_ok = np.abs(rr_ms - med_local) <= np.maximum(rel_thresh * med_local, abs_thresh_ms)
    drr = np.diff(rr_ms, prepend=rr_ms[0])
    global_med = np.median(rr_ms)
    jump_ok = np.abs(drr) <= (0.2 * global_med)
    mad_ok = mad_based_z_scores(rr_ms) <= mad_z_thresh
    return ~(phys_ok & local_ok & jump_ok & mad_ok)  # True=artifact

def interpolate_nn(rr_ms, art_mask):
    t = np.cumsum(rr_ms) / 1000.0
    valid = ~art_mask
    if valid.sum() < 3:
        return rr_ms.copy(), t, {"too_many_artifacts": True, "max_gap_s": None}
    t_valid = t[valid]
    rr_valid = rr_ms[valid]
    kind = 'linear' if len(t_valid) < 4 else 'cubic'
    f = interpolate.interp1d(t_valid, rr_valid, kind=kind, bounds_error=False)
    rr_nn = rr_ms.copy()
    rr_interp = f(t)
    rr_interp[:np.searchsorted(t, t_valid[0])] = rr_valid[0]
    rr_interp[np.searchsorted(t, t_valid[-1], side='right'):] = rr_valid[-1]
    rr_nn[art_mask] = rr_interp[art_mask]

    max_gap_s = 0.0
    if art_mask.any():
        idx = np.where(art_mask)[0]
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in groups:
            gap_ms = rr_ms[g].sum()
            max_gap_s = max(max_gap_s, gap_ms / 1000.0)
    flags = {"too_many_artifacts": False, "max_gap_s": max_gap_s}
    return rr_nn, t, flags

def time_domain_metrics(rr_nn_ms):
    rr = rr_nn_ms.astype(float)
    sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
    diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else np.nan
    return sdnn, rmssd

def frequency_domain_lfhf(rr_nn_ms, fs_resample=4.0):
    t = np.cumsum(rr_nn_ms) / 1000.0
    t_uniform = np.arange(t[0], t[-1], 1.0/fs_resample)
    kind = 'linear' if len(rr_nn_ms) < 4 else 'cubic'
    f_interp = interpolate.interp1d(t, rr_nn_ms, kind=kind, bounds_error=False, fill_value="extrapolate")
    rr_uniform = f_interp(t_uniform) - np.mean(rr_nn_ms)
    nperseg = min(256, len(rr_uniform))
    if nperseg < 64:
        return np.nan, np.nan, np.nan, None, None
    freqs, pxx = signal.welch(rr_uniform, fs=fs_resample, window='hann', nperseg=nperseg, noverlap=nperseg//2, detrend=False, scaling='density')
    
    def bandpower(f, p, fmin, fmax):
        mask = (f >= fmin) & (f < fmax)
        if not np.any(mask):
            return np.nan
        return np.trapz(p[mask], f[mask])  # np.trapz 사용
    
    lf = bandpower(freqs, pxx, 0.04, 0.15)
    hf = bandpower(freqs, pxx, 0.15, 0.40)
    lfhf = lf / hf if hf and hf > 0 else np.nan
    return lfhf, lf, hf, freqs, pxx

def quality_report(art_mask, flags, total_beats):
    corrected_pct = (art_mask.sum()/total_beats)*100
    notes = []
    if corrected_pct>10: notes.append("아티팩트 보정률 > 10%: 시간/주파수 지표 신뢰 낮음")
    elif corrected_pct>5: notes.append("아티팩트 보정률 > 5%: 주파수 영역 해석 주의")
    if flags.get("max_gap_s",0) >3: notes.append("단일 연속 누락 > 3초: PSD 왜곡 가능성")
    return corrected_pct, notes

def parse_uploaded(file) -> pd.DataFrame:
    content = file.read()
    for encoding in ['utf-8-sig','utf-8','latin1']:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
            break
        except:
            continue
    else:
        raise ValueError("파일 읽기 실패")
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    rr_col_candidates = ['rr','rr_ms','ibi']
    rr_col = next((c for c in rr_col_candidates if c in cols), None)
    if rr_col is None:
        if df.shape[1]==1: rr_col = df.columns[0]
        else: raise ValueError("RR 컬럼을 찾을 수 없음")
    return df[[rr_col]].rename(columns={rr_col:'rr'})

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="HRV(5분) 분석기", layout="wide")
st.title("5분 HRV 분석(고령자 연구용)")

with st.expander("설정 옵션", expanded=False):
    rr_min = st.number_input("생리적 RR 최소(ms)", value=300, step=50)
    rr_max = st.number_input("생리적 RR 최대(ms)", value=2000, step=50)
    rel_thresh = st.slider("로컬 중앙값 상대 임계(%)",5,50,20)/100
    abs_thresh_ms = st.number_input("로컬 절대 임계(ms)",200, step=10)
    window = st.number_input("로컬 중앙값 윈도우(비트)",11, step=2)
    mad_z_thresh = st.number_input("MAD z-score 임계",5.0, step=0.5)
    fs_resample = st.selectbox("PSD 재표본화 주파수(Hz)", [2.0,4.0], index=1)

uploaded = st.file_uploader("RR 파일 업로드(CSV/TXT, 단일파일)", type=["csv","txt"])
run = st.button("분석 실행")

if uploaded and run:
    try:
        df_raw = parse_uploaded(uploaded)
        rr_ms = infer_units_and_to_ms(df_raw['rr'])
        n_beats = len(rr_ms)
        st.write(f"총 비트 수: {n_beats}")
        total_time_s = rr_ms.sum()/1000
        st.write(f"측정 길이(추정): {total_time_s:.1f} 초")
        if total_time_s<240 or total_time_s>360:
            st.warning("권장 5분 범위 벗어남")

        art_mask = detect_artifacts(rr_ms, rel_thresh, abs_thresh_ms, window, mad_z_thresh)
        rr_nn, t_s, flags = interpolate_nn(rr_ms, art_mask)
        corrected_pct, notes = quality_report(art_mask, flags, n_beats)

        st.subheader("품질지표")
        st.write(f"아티팩트 보정률: {corrected_pct:.2f}%")
        st.write(f"최장 연속 누락: {flags.get('max_gap_s',0):.2f} s")
        for n in notes: st.error(n)

        sdnn, rmssd = time_domain_metrics(rr_nn)
        lfhf, lf, hf, freqs, pxx = frequency_domain_lfhf(rr_nn, fs_resample)

        st.subheader("지표 결과")
        c1,c2,c3 = st.columns(3)
        c1.metric("SDNN (ms)",f"{sdnn:.2f}" if not np.isnan(sdnn) else "NaN")
        c2.metric("RMSSD (ms)",f"{rmssd:.2f}" if not np.isnan(rmssd) else "NaN")
        c3.metric("LF/HF",f"{lfhf:.2f}" if not np.isnan(lfhf) else "NaN")
        st.write(f"LF:{lf:.2f}, HF:{hf:.2f}")

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(n_beats), y=rr_ms, mode='lines+markers', name='원시 RR', line=dict(color='lightgray')))
        if art_mask.any(): fig.add_trace(go.Scatter(x=np.where(art_mask)[0], y=rr_ms[art_mask], mode='markers', name='아티팩트', marker=dict(color='red', size=6)))
        fig.add_trace(go.Scatter(x=np.arange(n_beats), y=rr_nn, mode='lines', name='보정 RR', line=dict(color='blue')))
        fig.update_layout(title="RR 타코그램", xaxis_title="Beat Index", yaxis_title="RR(ms)")
        st.plotly_chart(fig, use_container_width=True)

        if freqs is not None and pxx is not None:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=freqs, y=pxx, mode='lines', name='PSD'))
            fig2.add_vrect(x0=0.04,x1=0.15,fillcolor="orange",opacity=0.2,line_width=0,annotation_text="LF")
            fig2.add_vrect(x0=0.15,x1=0.40,fillcolor="green",opacity=0.2,line_width=0,annotation_text="HF")
            fig2.update_layout(title="Welch PSD", xaxis_title="Frequency(Hz)", yaxis_title="Power Density")
            st.plotly_chart(fig2,use_container_width=True)
        else: st.warning("PSD 계산에 충분한 포인트 없음")

        out = pd.DataFrame({
            "sdnn_ms":[sdnn],"rmssd_ms":[rmssd],"lf_power":[lf],"hf_power":[hf],
            "lf_hf_ratio":[lfhf],"corrected_pct":[corrected_pct],
            "max_gap_s":[flags.get("max_gap_s",np.nan)],"total_time_s":[total_time_s],"n_beats":[n_beats]
        })
        st.download_button("결과 CSV 다운로드",data=out.to_csv(index=False).encode('utf-8-sig'),file_name="hrv_results.csv",mime="text/csv")
    except Exception as e:
        st.error(f"분석 중 오류: {str(e)}")
