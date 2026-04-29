#!/usr/bin/env python3
"""
Pilot Experiment — Audio Stimulus Generator v5 (BRIR 버전 + 레퍼런스)
9 conditions: ITDG (0/20/40ms) x Level Ratio (-6/0/+6dB)
+ reference.mp3: 경로A (정면 BRIR) 만 — PA 없는 원음 기준
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import sofar
import os, subprocess, tempfile

# ─── SETTINGS ──────────────────────────────────────────────────────────
INPUT_FILE = "Soundiron_VORT_Phrase_Latin_100BPM_E_27.wav"
OUTPUT_DIR = "audio"

BRIR_FILE = "/Users/user/Downloads/pilot-experiment/360-BRIR-FOAIR-database/Binaural/SOFA/C8m.sofa"

AZ_A =   0.0   # 경로A: 정면
AZ_B =  21.6   # 경로B: 무대 좌측 클러스터 (~±20°)
AZ_C = 338.4   # 경로C: 무대 우측 클러스터 (360° - 21.6°)

CONDITIONS = [
    ( 0, -6), ( 0,  0), ( 0, +6),
    (20, -6), (20,  0), (20, +6),
    (40, -6), (40,  0), (40, +6),
]

# ─── 함수 ──────────────────────────────────────────────────────────────
def load_brir(sofa_path):
    sofa_obj = sofar.read_sofa(sofa_path)
    ir  = sofa_obj.Data_IR
    pos = sofa_obj.SourcePosition
    sr  = int(sofa_obj.Data_SamplingRate)
    print(f"BRIR loaded: {os.path.basename(sofa_path)}")
    print(f"  SR={sr}Hz | Measurements={ir.shape[0]} | IR length={ir.shape[2]} samples")
    return ir, pos, sr

def get_brir_at(ir, pos, target_az, target_el=0.0):
    az = pos[:, 0]
    el = pos[:, 1]
    diffs = np.abs(az - target_az)
    diffs = np.minimum(diffs, 360.0 - diffs)
    el_mask = np.abs(el - target_el) < 5.0
    if np.any(el_mask):
        idx = np.argmin(np.where(el_mask, diffs, np.inf))
    else:
        idx = np.argmin(diffs)
    print(f"  → 요청 {target_az}° | 실제 {pos[idx,0]:.1f}° (idx={idx})")
    return ir[idx]

def binaural_convolve(mono, brir):
    L = fftconvolve(mono, brir[0])[:len(mono)]
    R = fftconvolve(mono, brir[1])[:len(mono)]
    return np.stack([L, R], axis=1)

def normalize(sig, peak=0.9):
    m = np.max(np.abs(sig))
    return sig / m * peak if m > 0 else sig

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
    if sr_wav != sr_brir:
        print(f"[경고] 음원 SR({sr_wav}) ≠ BRIR SR({sr_brir})")
        print("      librosa.resample()로 리샘플링 필요")
        return
    sr = sr_wav

    print("\nBRIR 방향 매핑:")
    brir_A = get_brir_at(brir_all, pos, AZ_A)
    brir_B = get_brir_at(brir_all, pos, AZ_B)
    brir_C = get_brir_at(brir_all, pos, AZ_C)
    print(f"출력: MP3 320kbps → ./{OUTPUT_DIR}/\n")

    # ── 레퍼런스 생성: 경로A (정면 BRIR) 만 ───────────────────────────
    ref = binaural_convolve(vocal, brir_A)   # [N, 2]
    ref = normalize(ref)
    ref_int16 = (ref * 32767).astype(np.int16)
    ref_path  = os.path.join(OUTPUT_DIR, "reference.mp3")
    save_as_mp3(ref_int16, sr, ref_path)
    print(f"  [REF] reference.mp3  (경로A 정면 BRIR만 — PA 없음)")

    # ── 9개 조건 생성 ────────────────────────────────────────────────
    for i, (itdg_ms, level_db) in enumerate(CONDITIONS, 1):
        delay_smp = int(sr * itdg_ms / 1000)
        gain_b    = 10 ** (level_db / 20.0)

        path_a = binaural_convolve(vocal, brir_A)

        delayed = np.zeros_like(vocal)
        if delay_smp < len(vocal):
            delayed[delay_smp:] = vocal[:len(vocal) - delay_smp] * gain_b

        path_b = binaural_convolve(delayed, brir_B)
        path_c = binaural_convolve(delayed, brir_C)

        mix = path_a + path_b + path_c
        mix = normalize(mix)
        out_int16 = (mix * 32767).astype(np.int16)

        lv_str = f"+{level_db}dB" if level_db >= 0 else f"{level_db}dB"
        fname  = f"c{i:02d}_itdg{itdg_ms}ms_lv{lv_str}.mp3"
        save_as_mp3(out_int16, sr, os.path.join(OUTPUT_DIR, fname))
        print(f"  [{i}/9] {fname}  (ITDG={itdg_ms}ms, Level={lv_str})")

    print(f"\n완료! reference.mp3 + 9개 MP3 → ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    np.random.seed(42)
    main()
