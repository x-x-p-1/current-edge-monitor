"""数据发生器数值验证脚本（运行：python 07_测试工具/verify_simulator.py）"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_数据生成与仿真"))
from current_simulator import generate_dataset, Fault

sys.stdout.reconfigure(errors="replace")  # 兼容 Windows GBK 控制台（emoji→?）

FS = 16000.0

# 1) 正常三相：RMS 应≈0.71（基波 0.707 + 谐波/PWM 略高），A/B 相位差≈120°
sig, ts, meta = generate_dataset(duration=2.0, f1=50.0)
rms = np.sqrt(np.mean(sig ** 2, axis=0))
print("RMS/相:", np.round(rms, 4), " 预期≈0.71(含谐波略高)")
N = len(sig)
bin50 = int(round(50.0 / (FS / N)))  # 50Hz 对应 FFT bin
pa = np.angle(np.fft.rfft(sig[:, 0])[bin50])
pb = np.angle(np.fft.rfft(sig[:, 1])[bin50])
d = (pa - pb) % (2 * np.pi)
print("A-B相位差:", round(d, 3), "rad, 预期≈2.094(120°)")

# 2) 堵转：纯正常段(0-2s) vs 堵转段(2-3s) RMS 比值≈depth=1.8
sig2, _, _ = generate_dataset(
    duration=4.0, faults=[Fault(kind="stall", start=2.0, dur=1.0, depth=1.8)]
)
rms_n = np.sqrt(np.mean(sig2[0:32000, 0] ** 2))
rms_s = np.sqrt(np.mean(sig2[32000:48000, 0] ** 2))
print("正常段RMS:", round(rms_n, 4), " 堵转段RMS:", round(rms_s, 4),
      " 比值:", round(rms_s / rms_n, 3), "预期≈1.8")

# 3) 转子条边带：f1=50, s=0.03 -> 边带 47Hz / 53Hz 应可见（整段频谱）
sig3, _, _ = generate_dataset(
    duration=4.0, f1=50.0,
    faults=[Fault(kind="rotor_sideband", start=1.0, dur=2.0, slip=0.03, depth=0.05)],
)
freq = np.fft.rfftfreq(len(sig3), 1.0 / FS)
spec = np.abs(np.fft.rfft(sig3[:, 0]))
band = lambda lo, hi: float(spec[(freq >= lo) & (freq <= hi)].max())
print("50Hz基波:", round(band(48, 52), 2),
      " 47Hz边带:", round(band(45.5, 48.5), 4),
      " 53Hz边带:", round(band(51.5, 54.5), 4))
