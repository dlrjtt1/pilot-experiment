#!/usr/bin/env python3
"""
Pilot Experiment — Audio Stimulus Generator v4 (BRIR 버전)
9 conditions: ITDG (0/20/40ms) x Level Ratio (-6/0/+6dB)

[준비사항]
1. SOFA 파일 다운로드:
   https://zenodo.org/records/2641166
   → "360-BRIR-FOAIR-database.zip" 다운로드 (6.6 GB)
   → 압축 해제 후 Binaural/ 폴더 안의 .sofa 파일 경로를 아래 BRIR_FILE 에 입력

2. 라이브러리 설치:
   pip install sofar
   brew install ffmpeg

[신호 구조]
  경로A = dry vocal + BRIR(정면 0°)     → 무대 원음
  경로B = delayed vocal + BRIR(좌 -60°) → 좌측 PA 스피커
  경로C = delayed vocal + BRIR(우 +60°) → 우측 PA 스피커
  최종  = A + B + C (바이노럴 합산)
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import sofar
import os, subprocess, tempfile

# ─── SETTINGS ──────────────────────────────────────────────────────────
INPUT_FILE  = "Soundiron_VORT_Phrase_Latin_100BPM_E_27.wav"  # ← 원본 파일명
OUTPUT_DIR  = "audio"

# Zenodo에서 다운받은 SOFA 파일 경로 (압축 해제 후)
# 예: "360-BRIR-FOAIR-database/Binaural/C_1m.sofa"  (무대에서 1m 거리 센터)
BRIR_FILE   = "/Users/user/Downloads/pilot-experiment/360-BRIR-FOAIR-database/Binaural/SOFA/C8m.sofa"  # ← 경로 확인 후 수정

# 각 경로의 방위각 (azimuth) 설정
AZ_A =   0.0   # 경로A: 정면 (무대 원음)
AZ_B =  60.0   # 경로B: 좌측 60° (sofar은 0-360 기준, 315=-45)
AZ_C =  300.0   # 경로C: 우측 60°

CONDITIONS = [
    ( 0, -6), ( 0,  0), ( 0, +6),
    (20, -6), (20,  0), (20, +6),
    (40, -6), (40,  0), (40, +6),
]

# ─── BRIR 로드 ──────────────────────────────────────────────────────────
def load_brir(sofa_path):
    """SOFA 파일에서 BRIR 데이터 로드"""
    sofa_obj = sofar.read_sofa(sofa_path)
    ir   = sofa_obj.Data_IR          # [M, 2, N] — M:측정수, 2:L/R, N:샘플
    pos  = sofa_obj.SourcePosition   # [M, 3] — azimuth, elevation, distance
    sr   = int(sofa_obj.Data_SamplingRate)
    print(f"BRIR loaded: {os.path.basename(sofa_path)}")
    print(f"  SR={sr}Hz | Measurements={ir.shape[0]} | IR length={ir.shape[2]} samples")
    return ir, pos, sr

def get_brir_at(ir, pos, target_az, target_el=0.0):
    """목표 방위각에 가장 가까운 BRIR 반환 [2, N]"""
    az = pos[:, 0]
    el = pos[:, 1]
    # 방위각 거리 계산 (360° wrap-around 고려)
    diffs = np.abs(az - target_az)
    diffs = np.minimum(diffs, 360.0 - diffs)
    # elevation 일치하는 것 우선
    el_mask = np.abs(el - target_el) < 5.0
    if np.any(el_mask):
        diffs_masked = np.where(el_mask, diffs, np.inf)
        idx = np.argmin(diffs_masked)
    else:
        idx = np.argmin(diffs)
    actual_az = pos[idx, 0]
    print(f"  → 요청 {target_az}° | 실제 {actual_az:.1f}° (idx={idx})")
    return ir[idx]  # [2, N]

# ─── 바이노럴 컨볼루션 ──────────────────────────────────────────────────
def binaural_convolve(mono, brir):
    """
    mono 신호를 BRIR로 컨볼루션 → [N, 2] 스테레오 바이노럴
    brir shape: [2, ir_len]
    """
    L = fftconvolve(mono, brir[0])[:len(mono)]
    R = fftconvolve(mono, brir[1])[:len(mono)]
    return np.stack([L, R], axis=1)  # [N, 2]

# ─── MP3 저장 ───────────────────────────────────────────────────────────
def save_as_mp3(int16_stereo, sr, mp3_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        wavfile.write(tmp_path, sr, int16_stereo)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-b:a", "320k", mp3_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    finally:
        os.remove(tmp_path)

# ─── MAIN ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 음원 로드
    sr_wav, data = wavfile.read(INPUT_FILE)
    if data.dtype == np.int16:
        vocal = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        vocal = data.astype(np.float32) / 2147483648.0
    else:
        vocal = data.astype(np.float32)
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=1)
    vocal = vocal / (np.max(np.abs(vocal)) + 1e-8) * 0.9
    print(f"\nLoaded: {INPUT_FILE} | SR={sr_wav}Hz | {len(vocal)/sr_wav:.1f}s")

    # BRIR 로드
    brir_all, pos, sr_brir = load_brir(BRIR_FILE)

    # 샘플레이트 확인
    if sr_wav != sr_brir:
        print(f"[경고] 음원 SR({sr_wav}) ≠ BRIR SR({sr_brir})")
        print("      음원을 BRIR SR로 리샘플링하세요: librosa.resample() 사용")
        return
    sr = sr_wav

    # 각 방향의 BRIR 추출
    print("\nBRIR 방향 매핑:")
    brir_A = get_brir_at(brir_all, pos, AZ_A)   # 정면
    brir_B = get_brir_at(brir_all, pos, AZ_B)   # 좌측
    brir_C = get_brir_at(brir_all, pos, AZ_C)   # 우측
    print(f"출력: MP3 320kbps → ./{OUTPUT_DIR}/\n")

    for i, (itdg_ms, level_db) in enumerate(CONDITIONS, 1):
        delay_smp = int(sr * itdg_ms / 1000)
        gain_b    = 10 ** (level_db / 20.0)

        # ── 경로A: 원음 → BRIR(정면) 컨볼루션 ────────────────────────
        path_a = binaural_convolve(vocal, brir_A)    # [N, 2]

        # ── 경로B/C: 지연 + 레벨 조정된 음원 ─────────────────────────
        delayed = np.zeros_like(vocal)
        if delay_smp < len(vocal):
            delayed[delay_smp:] = vocal[:len(vocal) - delay_smp] * gain_b

        path_b = binaural_convolve(delayed, brir_B)  # 좌측 [N, 2]
        path_c = binaural_convolve(delayed, brir_C)  # 우측 [N, 2]

        # ── 바이노럴 합산 ─────────────────────────────────────────────
        mix = path_a + path_b + path_c               # [N, 2]

        # ── 노말라이즈 ───────────────────────────────────────────────
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix = mix / peak * 0.9

        out_int16 = (mix * 32767).astype(np.int16)

        lv_str = f"+{level_db}dB" if level_db >= 0 else f"{level_db}dB"
        fname  = f"c{i:02d}_itdg{itdg_ms}ms_lv{lv_str}.mp3"
        fpath  = os.path.join(OUTPUT_DIR, fname)

        save_as_mp3(out_int16, sr, fpath)
        print(f"  [{i}/9] {fname}  (ITDG={itdg_ms}ms, Level={lv_str})")

    print(f"\n완료! 9개 MP3 → ./{OUTPUT_DIR}/")
    print("\n[사용 BRIR 정보]")
    print(f"  파일  : {BRIR_FILE}")
    print(f"  경로A : {AZ_A}° (정면)")
    print(f"  경로B : {AZ_B}° (좌측 -60°)")
    print(f"  경로C : {AZ_C}° (우측 +60°)")

if __name__ == "__main__":
    np.random.seed(42)
    main()
