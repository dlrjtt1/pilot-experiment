# 임시 테스트 — BRIR 방향이 실제로 다르게 들리는지 확인
import sofar, numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

sofa = sofar.read_sofa("/Users/user/Downloads/pilot-experiment/360-BRIR-FOAIR-database/Binaural/SOFA/C4m.sofa")
ir = sofa.Data_IR
pos = sofa.SourcePosition
sr  = int(sofa.Data_SamplingRate)

# 정면(0°) BRIR만 적용한 파일 저장
sr_in, data = wavfile.read("/Users/user/Downloads/pilot-experiment/Soundiron_VORT_Phrase_Latin_100BPM_E_27.wav")
vocal = data.astype(np.float32) / 32768.0
if vocal.ndim == 2: vocal = vocal.mean(axis=1)

# 정면 BRIR
idx_front = np.argmin(np.abs(pos[:,0] - 0.0))
brir_front = ir[idx_front]
L = fftconvolve(vocal, brir_front[0])[:len(vocal)]
R = fftconvolve(vocal, brir_front[1])[:len(vocal)]
out = np.stack([L, R], axis=1)
out = (out / np.max(np.abs(out)) * 0.9 * 32767).astype(np.int16)
wavfile.write("test_front.wav", sr, out)

# 좌측(315°) BRIR
idx_left = np.argmin(np.abs(pos[:,0] - 315.0))
brir_left = ir[idx_left]
L = fftconvolve(vocal, brir_left[0])[:len(vocal)]
R = fftconvolve(vocal, brir_left[1])[:len(vocal)]
out = np.stack([L, R], axis=1)
out = (out / np.max(np.abs(out)) * 0.9 * 32767).astype(np.int16)
wavfile.write("test_left.wav", sr, out)

print("test_front.wav 과 test_left.wav 를 헤드폰으로 비교해보세요")
print(f"사용된 방위각: front={pos[idx_front,0]:.1f}°, left={pos[idx_left,0]:.1f}°")