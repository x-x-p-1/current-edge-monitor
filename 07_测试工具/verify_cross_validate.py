"""
电流-电压交叉验证演示（运行：python 07_测试工具/verify_cross_validate.py）
========================================================================
原理：电流与电压通过电机阻抗 Z=V/I、功率因数 cosφ 物理耦合。同时测两者
可区分"供电侧问题"(电压先变) vs "电机/负载侧问题"(电流先变)，提早定位根因。

场景（5kW 电机，满载）：
  [1] 正常满载   → V≈220V 相, I≈10.5A, Z≈21Ω, cosφ≈0.83
  [2] 堵转       → V 不变, I 激增 63A, Z 骤降 → 电机侧
  [3] 电压跌落15% → V 与 I 同步降, Z 稳定 → 供电侧
  [4] 电流不平衡 → I_unbalance 大而 V 平衡 → 电机侧不对称
"""
import os
import sys
import importlib

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "00_数据生成与仿真"))
sys.stdout.reconfigure(errors="replace")

_sim = importlib.import_module("current_simulator")
gen_vi, Fault, Motor = _sim.generate_vi_dataset, _sim.Fault, _sim.Motor
_cv = importlib.import_module("02_特征提取.cross_validate")

FS = 16000.0
motor = Motor()
REF = None


def _slice(x, a, b):
    i0, i1 = int(a * FS), int(b * FS)
    return x[i0:i1]


def _show(tag, cur, vol, a=1.0, b=3.0):
    global REF
    r = _cv.cross_report(_slice(vol, a, b), _slice(cur, a, b), FS, 50.0)
    verdict = _cv.classify_side(r, REF)
    if REF is None:
        REF = r
    print(f"[{tag}]  V={r['v_rms']:.0f}V  I={r['i_rms']:5.1f}A  "
          f"Z={r['z_ohm']:5.1f}Ω  cosφ={r['pf']:.2f}  "
          f"V不平衡={r['v_unbalance_pct']:.1f}%  I不平衡={r['i_unbalance_pct']:.1f}%")
    print(f"         → {verdict}")


print("=" * 76)
print("电流-电压交叉验证 — 供电侧 vs 电机侧 判别演示（5kW 电机）")
print("=" * 76)

# [1] 正常满载（作为参考基线）
cur, vol, _, _ = gen_vi(duration=4.0, motor=motor, load_profile=[(0, 4, 1.0)])
print("\n[1] 正常满载：")
_show("1", cur, vol)

# [2] 堵转：电流激增，电压不变 → 阻抗骤降 → 电机侧
cur2, vol2, _, _ = gen_vi(
    duration=4.0, motor=motor, load_profile=[(0, 4, 1.0)],
    faults=[Fault(kind="stall", start=1.5, dur=1.5)])
print("\n[2] 堵转（电机侧故障）：")
_show("2", cur2, vol2, a=2.0, b=3.0)

# [3] 电压跌落 15%：V 与 I 同步降，阻抗稳定 → 供电侧
cur3, vol3, _, _ = gen_vi(
    duration=4.0, motor=motor, load_profile=[(0, 4, 1.0)], voltage_scale=0.85)
print("\n[3] 电压跌落 15%（供电侧问题）：")
_show("3", cur3, vol3)

# [4] 电流不平衡（电机不对称）：I 不平衡大而 V 平衡
cur4, vol4, _, _ = gen_vi(
    duration=4.0, motor=motor, load_profile=[(0, 4, 1.0)],
    faults=[Fault(kind="unbalance", start=0.5, dur=3.0, depth=0.08)])
print("\n[4] 电流不平衡 8%（电机侧不对称）：")
_show("4", cur4, vol4)

print("\n" + "=" * 76)
print("结论：仅测电流只能看到'电流异常'；同时测电压→可定位是供电侧还是电机侧，")
print("      从而提早、准确地干预。下一步可并入快/慢路径做实时交叉验证报警。")
print("=" * 76)
