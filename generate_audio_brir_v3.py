#!/usr/bin/env python3
"""
Pilot Experiment - Audio Stimulus Generator v3 (BRIR + LUFS)
9 conditions: ITDG (0/20/40ms) x Level Ratio (-6/0/+6dB)
+ reference.mp3: path A only (no PA)

v3 changes:
  - AZ_B/C: 60/300 -> 21.6/338.4 (concert hall main cluster angle)
  - normalization: peak -> -23 LUFS (ITU-R BS.1770)
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyloudnorm as pyln
import sofar
import os, subprocess, tempfile

# --- SETTINGS ---
INPUT_FILE  = "Soundiron_VORT_Phrase_Latin_100BPM_E_27.wav"
OUTPUT_DIR  = "audio"
BRIR_FILE   = "/Users/user/Downloads/pilot-experiment/360-BRIR-FOAIR-database/Binaural/SOFA/C8m.sofa"

TARGET_LUFS = -23.0  # ITU-R BS.1770

AZ_A = 0.0    # path A: front (stage)
AZ_B = 21.6   # path B: left cluster
AZ_C = 338.4  # path C: right cluster

CONDITIONS = [
    ( 0, -6), ( 0,  0), ( 0, +6),
    (20, -6), (20,  0), (20, +6),
    (40, -6), (40,  0), (40, +6),
]

# --- FUNCTIONS ---
def load_brir(sofa_path):
    obj = sofar.read_sofa(sofa_path)
    ir  = obj.Data_IR
    pos = obj.SourcePosition
    sr  = int(obj.Data_SamplingRate)
    print("BRIR: " + os.path.basename(sofa_path))
    print("  SR=" + str(sr) + "Hz | Measurements=" + str(ir.shape[0]) + " | IR=" + str(ir.shape[2]) + " samples")
    return ir, pos, sr

def get_brir_at(ir, pos, target_az, target_el=0.0):
    az = pos[:, 0]
    el = pos[:, 1]
    diffs = np.abs(az - target_az)
    diffs = np.minimum(diffs, 360.0 - diffs)
    el_mask = np.abs(el - target_el) < 5.0
    idx = np.argmin(np.where(el_mask, diffs, np.inf)) if np.any(el_mask) else np.argmin(diffs)
    print("  -> req " + str(target_az) + "deg | actual " + str(round(pos[idx,0],1)) + "deg (idx=" + str(idx) + ")")
    return ir[idx]

def binaural_convolve(mono, brir):
    L = fftconvolve(mono, brir[0])[:len(mono)]
    R = fftconvolve(mono, brir[1])[:len(mono)]
    return np.stack([L, R], axis=1)

def lufs_normalize(sig, sr, target=TARGET_LUFS):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(sig)
        if not np.isfinite(loudness) or loudness < -70:
            raise ValueError("silence")
        out = pyln.normalize.loudness(sig, loudness, target)
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out = out / peak * 0.9
        return out
    except Exception as e:
        print("  [warn] LUFS failed (" + str(e) + "), using peak normalize")
        m = np.max(np.abs(sig))
        return sig / m * 0.9 if m > 0 else sig

def save_mp3(stereo_float, sr, path):
    arr = np.clip(stereo_float, -1.0, 1.0)
    arr = (arr * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        wavfile.write(tmp_path, sr, arr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-b:a", "320k", path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    finally:
        os.remove(tmp_path)

# --- MAIN ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load source
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
    print("Loaded: " + INPUT_FILE + " | SR=" + str(sr_wav) + "Hz | " + str(round(len(vocal)/sr_wav,1)) + "s")

    # load BRIR
    brir_all, pos, sr_brir = load_brir(BRIR_FILE)
    if sr_wav != sr_brir:
        print("[ERROR] SR mismatch: source=" + str(sr_wav) + " BRIR=" + str(sr_brir))
        print("  -> use librosa.resample() to fix")
        return
    sr = sr_wav

    print("\nBRIR angle mapping:")
    brir_A = get_brir_at(brir_all, pos, AZ_A)
    brir_B = get_brir_at(brir_all, pos, AZ_B)
    brir_C = get_brir_at(brir_all, pos, AZ_C)
    print("\nOutput: MP3 320kbps -> ./" + OUTPUT_DIR + "/")
    print("Normalization: " + str(TARGET_LUFS) + " LUFS (ITU-R BS.1770)\n")

    # reference: path A only
    ref = binaural_convolve(vocal, brir_A)
    ref = lufs_normalize(ref, sr)
    save_mp3(ref, sr, os.path.join(OUTPUT_DIR, "reference.mp3"))
    print("  [REF] reference.mp3  (path A only, no PA, " + str(TARGET_LUFS) + " LUFS)")

    # 9 conditions
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
        mix = lufs_normalize(mix, sr)

        lv_str = ("+" if level_db >= 0 else "") + str(level_db) + "dB"
        fname  = "c" + str(i).zfill(2) + "_itdg" + str(itdg_ms) + "ms_lv" + lv_str + ".mp3"
        save_mp3(mix, sr, os.path.join(OUTPUT_DIR, fname))
        print("  [" + str(i) + "/9] " + fname + "  (ITDG=" + str(itdg_ms) + "ms, Level=" + lv_str + ", " + str(TARGET_LUFS) + " LUFS)")

    print("\nDone! reference.mp3 + 9 MP3 -> ./" + OUTPUT_DIR + "/")
    print("All files normalized to " + str(TARGET_LUFS) + " LUFS")

if __name__ == "__main__":
    np.random.seed(42)
    main()