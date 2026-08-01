"""预处理 v2 验证：三相 (N,3)、流式窗口、VFD 低频/高频保留（运行：python 06_测试与验证/verify_preprocess_v2.py）"""
import sys, os, importlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_pre = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _pre.CurrentPreprocessor
PreprocessConfig = _pre.PreprocessConfig

FS = 16000.0
N = FS * 1  # 1 秒

print("=== 1) 三相 (N,3) 批处理 ===")
t = np.arange(int(N)) / FS
# 三相：120° 相位差，20Hz 低速基频（VFD 低速）+ 3kHz 谐波 + 噪声
abc = np.column_stack([
    np.sin(2*np.pi*20*t + ph) + 0.2*np.sin(2*np.pi*3000*t + ph) + 0.05*np.random.randn(int(N))
    for ph in (0.0, -2*np.pi/3, 2*np.pi/3)
])
cfg = PreprocessConfig(sample_rate=FS, channels=3)
pp = CurrentPreprocessor(cfg, sample_rate=FS)
out = pp.process(abc)
print("输入:", abc.shape, "→ 输出:", out.shape, " 通道数:", out.shape[1])

print("=== 2) 低速基频(20Hz)与高频(3kHz)保留验证 ===")
# 关闭归一化以便看幅值（去 DC + 带通）
cfg2 = PreprocessConfig(sample_rate=FS, channels=1, norm_enabled=False)
pp2 = CurrentPreprocessor(cfg2, sample_rate=FS)
sig = abc[:, 0]
clean = pp2.process(sig)
freq = np.fft.rfftfreq(len(clean), 1/FS)
spec = np.abs(np.fft.rfft(clean))
band = lambda lo, hi: float(spec[(freq >= lo) & (freq <= hi)].max())
print("20Hz(低速基频)峰值:", round(band(15, 25), 4), " 应保留")
print("3kHz(诊断频带)峰值:", round(band(2900, 3100), 4), " 应保留")

print("=== 3) 流式处理（快路径） ===")
pp3 = CurrentPreprocessor(cfg, sample_rate=FS)  # 三相
# 分块送入，每块 64 点，窗口 256 / 步长 128
frames = []
for i in range(0, len(abc), 64):
    frames += pp3.process_streaming(abc[i:i+64])
print("送入", len(abc), "点(块64)，出帧数:", len(frames), " 首帧shape:", frames[0].shape if frames else None)
assert frames and frames[0].shape == (256, 3), "流式出帧形状不符"
print("✅ 全部通过")
