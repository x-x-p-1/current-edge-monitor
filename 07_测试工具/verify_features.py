"""特征提取 v2 验证：快/慢路径、三相特征、MCSA 边带（运行：python 07_测试工具/verify_features.py）"""
import sys, os, importlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")  # 兼容 Windows GBK 控制台（emoji→?）
_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
Fault = _sim.Fault
_feat = importlib.import_module("02_特征提取")

FS = 16000.0

print("=== 1) 快路径特征（短窗 256 点 @16k ≈ 16ms，三相） ===")
sig, ts, meta = generate_dataset(duration=1.0, f1=50.0)
fast = _feat.extract_fast_features(sig[:256], FS)
print("快路径特征数:", len(fast), " 示例:", {k: round(v,4) for k,v in list(fast.items())[:6]})
assert "ch0_rms" in fast and "ch2_env_cv" in fast

print("=== 2) 慢路径特征（长窗 4s，三相 + MCSA + 三相跨相） ===")
sig_long, _, _ = generate_dataset(duration=4.0, f1=50.0)
slow = _feat.extract_slow_features(sig_long, FS, f1=50.0)
n3p = sum(1 for k in slow if k.startswith("3p_"))
print("慢路径特征数:", len(slow), " 三相特征数:", n3p)
print("示例:", {k: round(v,4) for k,v in list(slow.items())[:8]})
assert n3p >= 10, "三相特征缺失"

print("=== 3) 三相特征数值验证（平衡 vs 不平衡） ===")
# 平衡三相：负序/零序≈0，相角误差≈0
bal = _feat.extract_three_phase_features(sig_long, FS, f1=50.0)
print("平衡: 负序比=%.4f 零序比=%.4f 相角误差max=%.4f" %
      (bal["3p_neg_ratio"], bal["3p_zero_ratio"], bal["3p_phase_err_max"]))
# 不平衡（B相减幅）
imb_sig = sig_long.copy(); imb_sig[:,1] *= 0.7
imb = _feat.extract_three_phase_features(imb_sig, FS, f1=50.0)
print("不平衡: 不平衡度=%.2f%% 负序比=%.4f" % (imb["3p_unbalance_pct"], imb["3p_neg_ratio"]))
assert bal["3p_neg_ratio"] < 0.05, "平衡时负序应≈0"
assert imb["3p_neg_ratio"] > bal["3p_neg_ratio"], "不平衡应增大负序"

print("=== 4) MCSA 边带特征验证（注入转子条边带） ===")
norm = _feat.compute_sideband_energy(sig_long[:,0], FS, f1=50.0, slip=0.03)
faulty, _, _ = generate_dataset(duration=4.0, f1=50.0,
    faults=[Fault(kind="rotor_sideband", start=1.0, dur=2.0, slip=0.03, depth=0.05)])
fb = _feat.compute_sideband_energy(faulty[:,0], FS, f1=50.0, slip=0.03)
print("正常: 边带比=%.5f | 注入边带: 边带比=%.5f" % (norm["sideband_ratio"], fb["sideband_ratio"]))
assert fb["sideband_ratio"] > norm["sideband_ratio"] * 5, "边带注入应显著提升边带能量比"

print("✅ 02 v2 全部通过")
