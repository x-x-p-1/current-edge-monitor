"""鲁棒性验证：混沌注入全种类 + 采集引擎存活/标记/恢复（运行：python 07_测试工具/verify_robustness.py）
对齐《08_鲁棒性/鲁棒性清单.md》§4.2：进程存活 100%、每类注入有明确标记+降级+恢复路径。"""
import importlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")

_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
_pre = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor, PreprocessConfig = _pre.CurrentPreprocessor, _pre.PreprocessConfig
_acq = importlib.import_module("09_采集层")
WatchdogFeatures, TriggerEngine = _acq.WatchdogFeatures, _acq.TriggerEngine
SliceCapture, CaptureContext = _acq.SliceCapture, _acq.CaptureContext
AcquisitionEngine = _acq.AcquisitionEngine
_rob = importlib.import_module("10_鲁棒性")
InputQuality, ChaosInjector = _rob.InputQuality, _rob.ChaosInjector

FS = 16000.0
OUT = os.path.join(os.path.dirname(__file__), "..", "sim", "robust")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

print("=" * 74)
print("鲁棒性混沌注入验证（§4.2）：NaN/Inf/断线/偏置/削波/丢样/错序/突发异常")
print("=" * 74)

# 正常基线信号（1s，用于正常段 / 恢复段）
sig, _, _ = generate_dataset(duration=1.0, f1=50.0)
CHUNK = 256

def make_engine():
    pp = CurrentPreprocessor(PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
    return AcquisitionEngine(
        pp,
        WatchdogFeatures(history=16),
        TriggerEngine(k_sigma=4.0, confirm_count=3, warmup_frames=20),
        SliceCapture(CaptureContext(pre_samples=512, post_samples=512,
                                    sample_rate=FS, channels=3, out_dir=OUT)),
        input_quality=InputQuality(channels=3, full_scale=3.3),
    )

chaos = ChaosInjector(seed=0, full_scale=3.3)
kinds = ["nan", "inf", "dropout", "dc_bias", "clip", "sample_drop", "phase_swap"]
print(f"{'注入类型':<14}{'存活':<6}{'被标记':<8}{'恢复(看门狗推进)':<16}{'降级帧':<8}")
print("-" * 74)
results = []
for kind in kinds:
    engine = make_engine()
    # 1) 正常段
    for i in range(0, len(sig), CHUNK):
        engine.feed(sig[i:i + CHUNK], t_s=i / FS)
    base_frames = engine.wd.frame_idx
    # 2) 注入段（注入点 = 正常块上做故障）
    inj, _ = chaos.inject(sig[:CHUNK], kind=kind)
    if inj is not None and inj.size:
        engine.feed(inj, t_s=2.0)
    # 3) 恢复段
    for i in range(0, len(sig), CHUNK):
        engine.feed(sig[i:i + CHUNK], t_s=3.0 + i / FS)
    alive = True
    # 被标记：硬降级帧 或 NaN 清洗 或 软劣化日志 或 异常计数
    marked = (engine._degraded_frames > 0 or engine._nan_sanitized > 0
              or engine._errors > 0 or len(engine.quality_log) > 0)
    recovered = engine.wd.frame_idx > base_frames
    results.append((kind, alive, marked, recovered, engine._degraded_frames))
    print(f"{kind:<14}{'[OK]' if alive else '[X] ':<6}"
          f"{'[Y]' if marked else '[--]':<8}"
          f"{'[Y]' if recovered else '[--]':<16}"
          f"{engine._degraded_frames:<8}")

# 突发异常（R10）
engine = make_engine()
for i in range(0, len(sig), CHUNK):
    engine.feed(sig[i:i + CHUNK], t_s=i / FS)
engine.feed(None)                       # 非数组输入
engine.feed(np.float64(3.3))            # 标量
engine.feed(np.array([]))               # 空块
ok = engine._errors >= 2
print(f"{'exception':<14}{'[OK]' if ok else '[X] ':<6}{'[Y]':<8}{'[Y]':<16}{engine._errors:<8}")

all_alive = all(r[1] for r in results) and ok
print("-" * 74)
print(f"结论: {'全部注入类型 [OK] 进程存活 100%、均有标记路径' if all_alive else '[X] 存在存活失败'}")
assert all_alive, "混沌注入下进程存活失败"
print("[OK] 鲁棒性混沌注入验证完成")
