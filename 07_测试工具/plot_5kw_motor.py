"""
5kW 电机物理模式可视化（运行：python 07_测试工具/plot_5kw_motor.py）
====================================================================
生成 sim/motor_5kw_demo.png：
  上：10s 三相电流全貌（load_step / stall / unbalance 三故障段高亮）
  中：RMS 包络 + 空载/额定/堵转参考线
  下：空载 / 满载 / 堵转 三段波形放大（各 ~2 个工频周期）
"""
import os
import sys
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "00_数据生成与仿真"))
sys.stdout.reconfigure(errors="replace")

from current_simulator import generate_dataset, Fault, Motor

FS = 16000.0
motor = Motor()  # 5kW / 380V / 50Hz / 4 极

sig, ts, meta = generate_dataset(
    duration=10.0, f1=50.0, motor=motor,
    load_profile=[(0.0, 2.0, 0.0), (2.0, 7.0, 1.0),
                  (7.0, 9.0, 1.0), (9.0, 10.0, 0.0)],
    faults=[
        Fault(kind="load_step", start=2.0, dur=0.3, depth=1.0),
        Fault(kind="stall", start=4.0, dur=1.0),
        Fault(kind="unbalance", start=7.0, dur=2.0, depth=0.05),
    ],
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("matplotlib 未安装，无法出图。已生成数据，请检查 sim/abc_5kw.npy")
    np.save(os.path.join(_ROOT, "sim", "abc_5kw.npy"), sig)
    sys.exit(0)

# ── 降采样全貌（~1kHz 显示） ──
step = int(FS / 1000)
t_d = np.arange(len(sig))[::step] / FS
ia, ib, ic = sig[::step, 0], sig[::step, 1], sig[::step, 2]

# ── RMS 包络（100ms 窗） ──
win = int(0.1 * FS)
n_win = len(sig) // win
t_env = np.arange(n_win) * win / FS
env = np.array([
    np.sqrt(np.mean(sig[i * win:(i + 1) * win] ** 2, axis=0))
    for i in range(n_win)
])

SEGS = [(2, 3, "load_step 负载突变", "#9467bd"),
        (4, 5, "stall 堵转", "#d62728"),
        (7, 9, "unbalance 不平衡", "#8c564b")]

fig = plt.figure(figsize=(14, 11))
gs = GridSpec(3, 1, height_ratios=[2.2, 1.6, 1.6], figure=fig)

# 1) 三相电流全貌
ax = fig.add_subplot(gs[0])
ax.plot(t_d, ia, lw=0.5, color="#1f77b4", label="A相")
ax.plot(t_d, ib, lw=0.5, color="#ff7f0e", label="B相")
ax.plot(t_d, ic, lw=0.5, color="#2ca02c", label="C相")
ax.set_title("5kW 电机物理模式 — 三相电流（真实安培）", fontsize=13)
ax.set_ylabel("电流 (A)")
ax.set_xlim(0, 10)
ax.set_ylim(-90, 90)
for a, b, lbl, c in SEGS:
    ax.axvspan(a, b, color=c, alpha=0.15)
    ax.text((a + b) / 2, -84, lbl, ha="center", fontsize=9, color=c)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# 2) RMS 包络 + 参考线
ax = fig.add_subplot(gs[1])
ax.plot(t_env, env[:, 0], color="#1f77b4", label="A相 RMS")
ax.plot(t_env, env[:, 1], color="#ff7f0e", label="B相 RMS")
ax.plot(t_env, env[:, 2], color="#2ca02c", label="C相 RMS")
ax.axhline(motor.no_load_current_a, ls="--", color="gray", lw=1,
           label=f"空载 {motor.no_load_current_a:.1f}A")
ax.axhline(motor.rated_current_a, ls="--", color="black", lw=1,
           label=f"额定 {motor.rated_current_a:.1f}A")
ax.axhline(motor.stall_current_a, ls="--", color="red", lw=1,
           label=f"堵转 {motor.stall_current_a:.0f}A")
ax.set_ylabel("RMS (A)")
ax.set_xlabel("时间 (s)")
ax.set_xlim(0, 10)
ax.set_ylim(0, 75)
for a, b, lbl, c in SEGS:
    ax.axvspan(a, b, color=c, alpha=0.12)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# 3) 空载 / 满载 / 堵转 三段放大（各 ~2 个工频周期）
ax = fig.add_subplot(gs[2])
zseg = [(0.20, 0.24, "空载 (≈3.5A)", "#1f77b4"),
        (3.20, 3.24, "满载 (≈10.5A)", "#2ca02c"),
        (4.02, 4.06, "堵转 (≈63A)", "#d62728")]
for a, b, lbl, c in zseg:
    i0, i1 = int(a * FS), int(b * FS)
    tz = np.arange(i0, i1) / FS
    ax.plot(tz, sig[i0:i1, 0], lw=1.2, color=c,
            label=lbl + f"  (峰值 {np.max(np.abs(sig[i0:i1, 0])):.0f}A)")
ax.set_title("波形放大：空载 / 满载 / 堵转（A 相）", fontsize=12)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("电流 (A)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(_ROOT, "sim", "motor_5kw_demo.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=110)
print(f"已生成: {os.path.abspath(out)}")
print(f"单位: {meta['units']}  削波率: {meta['clip_ratio']:.4f}")
