"""M1 采集层验证：环形缓冲 + 看门狗 + 触发 + 切片 端到端（运行：python 07_测试工具/verify_acquisition.py）
演示：仿真信号（正常→堵转→正常）→ 流式预处理 → 全速率环形缓冲 → 看门狗特征 → 触发引擎 → 切片落盘。
同时给出每帧处理耗时基准（对齐 TODO B：看门狗 ~1.8ms/帧 目标）。"""
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")

_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
Fault = _sim.Fault
_pre = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor, PreprocessConfig = _pre.CurrentPreprocessor, _pre.PreprocessConfig
_acq = importlib.import_module("09_采集层")
WatchdogFeatures = _acq.WatchdogFeatures
TriggerEngine = _acq.TriggerEngine
SliceCapture = _acq.SliceCapture
CaptureContext = _acq.CaptureContext
AcquisitionEngine = _acq.AcquisitionEngine

FS = 16000.0
OUT = os.path.join(os.path.dirname(__file__), "..", "sim", "slices")
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("M1 采集层端到端验证：环形缓冲 / 看门狗 / 触发 / 切片")
print("=" * 70)

# ── 1) 生成仿真信号：正常 2s → 堵转 1.5s → 正常 2s ──
sig, ts, meta = generate_dataset(
    duration=5.5, f1=50.0,
    faults=[Fault(kind="stall", start=2.0, dur=1.5, depth=2.0)],
)
print(f"[1] 信号: {sig.shape} ({sig.shape[0]/FS:.1f}s × {sig.shape[1]}相)")

# ── 2) 组装 M1 采集引擎（幅值保持预处理） ──
pp = CurrentPreprocessor(
    PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
wd = WatchdogFeatures(history=16)
trig = TriggerEngine(k_sigma=4.0, confirm_count=3, release_count=5, warmup_frames=30)
cap = SliceCapture(CaptureContext(
    pre_samples=2048, post_samples=2048, sample_rate=FS, channels=3, out_dir=OUT))
engine = AcquisitionEngine(pp, wd, trig, cap)

# ── 3) 流式喂入（块 512 点），计时 ──
t0 = time.perf_counter()
frame_times = []
n_frames = 0
saved = []
CHUNK = 512
for i in range(0, len(sig), CHUNK):
    t1 = time.perf_counter()
    res = engine.feed(sig[i:i + CHUNK], t_s=i / FS, meta={"phase": "verify"})
    frame_times.append((time.perf_counter() - t1) / max(len(res) or 1, 1))
    saved += res
elapsed = time.perf_counter() - t0

print(f"[2] 流式喂入: {len(sig)} 点（块 {CHUNK}）  耗时 {elapsed*1000:.1f}ms")
print(f"[3] 看门狗帧数: {wd.frame_idx}   触发激活态(末尾): {engine.trig.last_event.active}")
print(f"[4] 切片保存（=触发上升沿数）: {len(saved)} 个")

# ── 4) 切片内容检查 ──
if saved:
    info = saved[0]
    z = np.load(info["path"])
    data = z["data"]
    pre = int(z["pre_avail"])
    print(f"[5] 切片: {os.path.basename(info['path'])}")
    print(f"    shape={data.shape} 预触发={pre}/{int(z['pre_samples'])} "
          f"后触发={int(z['post_samples'])}  reason={str(z['reason'])}")
    # 切片 RMS 应明显高于正常（含堵转段）
    rms_slice = float(np.sqrt(np.mean(data ** 2)))
    print(f"    切片整体 RMS = {rms_slice:.3f}（正常基线 RMS≈0.71）")

# ── 5) 每帧耗时基准（含预处理+看门狗+触发） ──
per_frame = np.mean(frame_times)
p99 = np.percentile(frame_times, 99)
print(f"[6] 每块处理(预处理+看门狗+触发): 均值 {per_frame*1000:.2f}ms  块内每帧 ≈ {per_frame*1000/(CHUNK/128):.2f}ms/帧")
print(f"    目标: 看门狗 ~1.8ms/帧 → {'✅ 达标' if per_frame*1000/(CHUNK/128) < 1.8 else '⚠️ 超时'}")

print("✅ M1 采集层端到端验证完成")
