#!/usr/bin/env python3
"""
BRIR 내부 지연 측정 도구
각 방향(정면/좌/우) BRIR의 직접음 도달 시간을 측정해서
실험에 맞는 ITDG 보정값을 계산해줍니다.
"""

import numpy as np
import sofar
import sys

# ─── 설정 ──────────────────────────────────────────────────────────────
BRIR_FILE = "/Users/user/Downloads/pilot-experiment/360-BRIR-FOAIR-database/Binaural/SOFA/C8m.sofa"

AZ_A =   0.0   # 경로A: 정면
AZ_B =  60.0   # 경로B: 좌측 60°
AZ_C = 300.0   # 경로C: 우측 60° (= -60°)

THRESHOLD_DB = -20  # 직접음 감지 임계값 (dB) — 피크 대비

# ─── 함수 ──────────────────────────────────────────────────────────────
def get_idx(pos, target_az):
    az = pos[:, 0]
    diffs = np.abs(az - target_az)
    diffs = np.minimum(diffs, 360.0 - diffs)
    return np.argmin(diffs)

def find_onset(ir_mono, sr, threshold_db=-20):
    """
    직접음 onset 위치 찾기
    피크 대비 threshold_db 이상인 첫 샘플 위치 반환
    """
    peak = np.max(np.abs(ir_mono))
    threshold_linear = peak * 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(ir_mono) > threshold_linear)[0]
    if len(above) == 0:
        return 0
    return above[0]

# ─── 메인 ──────────────────────────────────────────────────────────────
sofa_obj = sofar.read_sofa(BRIR_FILE)
ir  = sofa_obj.Data_IR         # [M, 2, N]
pos = sofa_obj.SourcePosition  # [M, 3]
sr  = int(sofa_obj.Data_SamplingRate)

print(f"BRIR 파일  : {BRIR_FILE.split('/')[-1]}")
print(f"샘플레이트 : {sr} Hz")
print(f"측정 수    : {ir.shape[0]} directions")
print(f"IR 길이    : {ir.shape[2]} samples ({ir.shape[2]/sr*1000:.1f} ms)")
print()

results = {}
for label, az in [("경로A (정면)", AZ_A), ("경로B (좌측)", AZ_B), ("경로C (우측)", AZ_C)]:
    idx = get_idx(pos, az)
    actual_az = pos[idx, 0]

    # L채널 기준으로 onset 측정
    ir_L = ir[idx, 0, :]
    ir_R = ir[idx, 1, :]
    onset_L = find_onset(ir_L, sr, THRESHOLD_DB)
    onset_R = find_onset(ir_R, sr, THRESHOLD_DB)
    onset = min(onset_L, onset_R)

    delay_ms = onset / sr * 1000
    results[label] = {"onset": onset, "delay_ms": delay_ms, "az": actual_az}

    print(f"{label}")
    print(f"  요청 방위각   : {az}°  →  실제 {actual_az:.1f}°")
    print(f"  직접음 onset  : {onset} samples = {delay_ms:.2f} ms")
    print()

# ─── ITDG 보정값 계산 ──────────────────────────────────────────────────
onset_A = results["경로A (정면)"]["onset"]
onset_B = results["경로B (좌측)"]["onset"]
onset_C = results["경로C (우측)"]["onset"]

# B/C가 A보다 늦게 도착하는지 먼저 도착하는지
diff_B = (onset_B - onset_A) / sr * 1000  # ms, + 면 B가 늦음
diff_C = (onset_C - onset_A) / sr * 1000

print("=" * 50)
print("[BRIR 내부 지연 분석]")
print(f"  A(정면) vs B(좌측) : {diff_B:+.2f} ms  ({'B가 늦음' if diff_B > 0 else 'B가 빠름'})")
print(f"  A(정면) vs C(우측) : {diff_C:+.2f} ms  ({'C가 늦음' if diff_C > 0 else 'C가 빠름'})")
print()

# ─── 실험 ITDG 보정 권장값 ─────────────────────────────────────────────
print("[실험 ITDG 설정 권장값]")
print("  목표 ITDG = BRIR 내부 지연 차 + 추가 지연")
print()

for target_itdg in [0, 20, 40]:
    # 코드에서 추가해야 할 delay_samples
    # 실제 총 ITDG = BRIR 내부 지연 차 + 코드 딜레이
    # 우리가 원하는 총 ITDG = target_itdg
    # 따라서 코드 딜레이 = target_itdg - BRIR 내부 지연 차

    avg_brir_diff = (diff_B + diff_C) / 2  # B와 C 평균
    code_delay = target_itdg - avg_brir_diff

    if code_delay < 0:
        note = f"⚠️  A를 {abs(code_delay):.1f}ms 앞당겨야 함 (A pre-delay 필요)"
    else:
        note = f"→ 코드에서 B/C에 {code_delay:.1f}ms 딜레이 추가"

    print(f"  목표 ITDG {target_itdg:2d}ms : {note}")

print()
print("[참고]")
print("  선행음 효과 최적 범위 : 10~20ms (Haas, 1951)")
print("  에코 분리 시작점      : ~40ms")
