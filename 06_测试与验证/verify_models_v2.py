"""检测模型 v2 验证：过程状态识别（运行：python 06_测试与验证/verify_models_v2.py）"""
import sys, os, importlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
Fault = _sim.Fault
_feat = importlib.import_module("02_特征提取")
_mod = importlib.import_module("03_检测模型")
ProcessStateClassifier = _mod.ProcessStateClassifier
StateRuleConfig = _mod.StateRuleConfig

FS = 16000.0
WIN, STRIDE = 256, 128


def run_states(signal, clf):
    """逐帧快特征 → 状态机，返回状态序列"""
    states = []
    for i in range(0, len(signal) - WIN, STRIDE):
        win = signal[i:i + WIN]
        fast = _feat.extract_fast_features(win, FS)
        states.append(clf.update(fast)["state"])
    return states


print("=== 1) 负载周期：空载(0-2s) → 负载(2-8s) → 空载(8-10s) ===")
sig, _, _ = generate_dataset(
    duration=10.0, f1=50.0,
    load_profile=[(0.0, 2.0, 0.1), (2.0, 8.0, 1.0), (8.0, 10.0, 0.1)],
)
# 基线来自标定/P3：正常负载 RMS ≈ 0.71（模拟器 amp=1.0 的标幺值）
states = run_states(sig, ProcessStateClassifier(StateRuleConfig(initial_baseline=0.71)))
from collections import Counter
print("状态分布:", dict(Counter(states)))
# 中间时段应主要为 LOAD
mid = states[len(states)//5: len(states)//5 + len(states)//2]
assert Counter(mid).most_common(1)[0][0] == "LOAD", f"中段应为LOAD，实际 {Counter(mid).most_common(1)}"
print("✅ 负载周期状态识别正确")

print("=== 2) 堵转注入（5-6s，深度1.8）===")
sig2, _, _ = generate_dataset(
    duration=8.0, f1=50.0,
    load_profile=[(0.0, 8.0, 1.0)],
    faults=[Fault(kind="stall", start=5.0, dur=1.0, depth=1.8)],
)
clf = ProcessStateClassifier(StateRuleConfig(stall_confirm=3))
stall_seen = False
for i in range(0, len(sig2) - WIN, STRIDE):
    fast = _feat.extract_fast_features(sig2[i:i + WIN], FS)
    if clf.update(fast)["state"] == "STALL":
        stall_seen = True
        break
print("堵转检出:", stall_seen)
assert stall_seen, "堵转应被检出"
print("✅ 堵转检测正确")

print("=== 3) 停机段（幅值≈0）===")
sig3, _, _ = generate_dataset(duration=3.0, f1=50.0,
    load_profile=[(0.0, 3.0, 0.005)])
states3 = run_states(sig3, ProcessStateClassifier())
top3 = Counter(states3).most_common(1)[0][0]
print("主要状态:", top3)
assert top3 in ("STOP", "IDLE"), f"低幅值应 STOP/IDLE，实际 {top3}"
print("✅ 停机/空载识别正确")
