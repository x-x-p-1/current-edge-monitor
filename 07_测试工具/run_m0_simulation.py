"""
M0 端到端仿真（运行：python 00_数据生成与仿真/run_m0_simulation.py）
====================================================================
数据发生器 → 预处理(01) → 快路径(02) → 过程状态(03) → 迟滞/事件(04)
                                                              ↘ 慢路径(02) → MCSA/三相诊断

场景（30s，fs=16k，三相）：
  0–5s    启动/空载     (amp 0.15)
  5–20s   满载运行       (amp 1.0)
  10–14s  转子条边带故障  (MCSA，slip=0.03，depth=0.03)  ← 慢路径应检出
  20–22s  堵转           (×1.8)                          ← 状态机/事件应检出
  22–28s  恢复满载
  28–30s  停机           (amp 0.005)

输出：文本报告 + sim/m0_simulation.png（若装有 matplotlib）
"""
import os
import sys
import time
import importlib
from collections import Counter

import numpy as np

sys.stdout.reconfigure(errors="replace")  # 兼容 Windows GBK 控制台（emoji→?）

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "00_数据生成与仿真"))

_sim = importlib.import_module("current_simulator")
generate_dataset, Fault = _sim.generate_dataset, _sim.Fault
_pp = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _pp.CurrentPreprocessor
PreprocessConfig = _pp.PreprocessConfig
_feat = importlib.import_module("02_特征提取")
_ps = importlib.import_module("03_检测模型.process_state")
ProcessStateClassifier, StateRuleConfig = _ps.ProcessStateClassifier, _ps.StateRuleConfig
_hyst = importlib.import_module("04_后处理与决策.hysteresis")
HysteresisAlarm, EventAggregator, AlarmState = (_hyst.HysteresisAlarm,
                                                _hyst.EventAggregator, _hyst.AlarmState)

FS = 16000.0
WIN, STRIDE = 256, 128
BASE_LOAD_RMS = 0.71
STATE_ORDER = ["STOP", "TRANSIENT", "IDLE", "LOAD", "STALL", "UNKNOWN"]


def main():
    print("=" * 64)
    print("M0 端到端仿真 — 三相变频电机电流监测")
    print("=" * 64)

    # ── 1) 生成 30s 三相信号 ──
    t0 = time.perf_counter()
    sig, ts, meta = generate_dataset(
        duration=30.0, f1=50.0,
        load_profile=[(0.0, 5.0, 0.15), (5.0, 20.0, 1.0),
                      (20.0, 22.0, 1.0), (22.0, 28.0, 1.0), (28.0, 30.0, 0.005)],
        faults=[
            Fault(kind="rotor_sideband", start=10.0, dur=4.0, slip=0.03, depth=0.03),
            Fault(kind="stall", start=20.0, dur=2.0, depth=1.8),
        ],
    )
    print(f"[1] 信号生成: {sig.shape} ({sig.shape[0]/FS:.0f}s × {sig.shape[1]}相)  "
          f"削波={meta['clip_ratio']:.4f} 耗时={(time.perf_counter()-t0)*1000:.0f}ms")

    # ── 2) 快路径：预处理 + 快特征 + 过程状态机 ──
    pp = CurrentPreprocessor(
        PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
    clf = ProcessStateClassifier(StateRuleConfig(initial_baseline=BASE_LOAD_RMS,
                                                 stall_confirm=3))
    alarm = HysteresisAlarm(threshold_upper=0.6, threshold_lower=0.4,
                            confirm_count=3, release_count=3)
    agg = EventAggregator(merge_window_ms=500.0, frame_interval_ms=1000 * STRIDE / FS)

    states, times = [], []
    t_frames = []
    t0 = time.perf_counter()
    for i in range(0, len(sig) - WIN, STRIDE):
        f0 = time.perf_counter()
        win = pp.process(sig[i:i + WIN])
        fast = _feat.extract_fast_features(win, FS)
        res = clf.update(fast)
        t_frames.append((time.perf_counter() - f0) * 1e6)

        states.append(res["state"])
        times.append(i / FS)
        # 疑似异常 → 迟滞 → 事件
        score = 0.9 if res["state"] in ("STALL", "TRANSIENT") else 0.1
        st = alarm.update(score)
        agg.update(st == AlarmState.ALARM)
    wall = time.perf_counter() - t0
    frames = len(states)
    events = agg.finalize()

    print(f"[2] 快路径: {frames} 帧, 帧耗时 mean={np.mean(t_frames):.2f}µs "
          f"p99={np.percentile(t_frames,99):.2f}µs, 总耗={wall:.2f}s")
    dist = Counter(states)
    print("    状态分布:", dict(dist))
    print(f"    报警事件: {len(events)} 个 → {events}")

    # ── 3) 慢路径：每 2s 窗做频谱/MCSA/三相诊断 ──
    print("[3] 慢路径诊断（2s 窗）:")
    slow_report = []
    win_len = int(2.0 * FS)
    for ws in range(0, len(sig) - win_len + 1, win_len):
        seg = sig[ws:ws + win_len]
        # 停机段跳过（近零，频域/三相特征无意义）
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        slow = _feat.extract_slow_features(seg, FS, f1=50.0, slip=0.03)
        t_start = ws / FS
        slow_report.append({
            "t": t_start,
            "sideband_ratio": slow.get("ch0_sideband_ratio", 0.0),
            "thd": slow.get("ch0_thd", 0.0),
            "unbalance": slow.get("3p_unbalance_pct", 0.0),
            "neg_ratio": slow.get("3p_neg_ratio", 0.0),
        })
        print(f"    t={t_start:5.0f}s  sideband={slow['ch0_sideband_ratio']:.5f} "
              f"THD={slow['ch0_thd']*100:.2f}%  unbalance={slow['3p_unbalance_pct']:.1f}%")

    # ── 4) 关键结果断言/汇总 ──
    # 转子条边带：sideband_ratio 超过阈值(5e-4)且落入注入区间 8~16s
    sb_windows = [r["t"] for r in slow_report if r["sideband_ratio"] >= 5e-4]
    sb_detected = any(8 <= t <= 16 for t in sb_windows)
    print(f"[4] 转子条边带: 升高窗={sb_windows}s "
          f"({'✅ 检出(10-14s)' if sb_detected else '⚠️ 未检出'})")
    stall_frames = sum(1 for s in states if s == "STALL")
    print(f"    堵转帧数: {stall_frames} (注入 20–22s，"
          f"{'✅ 检出' if stall_frames > 0 else '⚠️ 未检出'})")

    # ── 5) 出图（可选） ──
    _plot(times, states, sig, slow_report)

    # ── 6) 写入 UTF-8 报告 ──
    import io
    lines = [
        "=" * 64,
        "M0 端到端仿真报告 — 三相变频电机电流监测",
        "=" * 64,
        f"信号: 30s × 3相 @16kSPS, 故障注入: 转子条边带(10-14s), 堵转(20-22s)",
        f"状态分布: {dict(dist)}",
        f"报警事件: {len(events)} 个 -> {events}",
        f"快路径帧耗时: mean={np.mean(t_frames):.2f}us p99={np.percentile(t_frames,99):.2f}us",
        "慢路径诊断(2s窗):",
    ]
    for r in slow_report:
        lines.append(
            f"  t={r['t']:5.0f}s sideband={r['sideband_ratio']:.5f} "
            f"THD={r['thd']*100:.2f}% unbalance={r['unbalance']:.1f}%"
        )
    lines.append(f"转子条边带: 升高窗={sb_windows}s ({'检出(10-14s)' if sb_detected else '未检出'})")
    lines.append(f"堵转帧数 {stall_frames} (注入20-22s, {'检出' if stall_frames>0 else '未检出'})")
    os.makedirs("sim", exist_ok=True)
    with open(os.path.join("sim", "m0_report_utf8.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[6] 已写入 sim/m0_report_utf8.txt")

    print("=" * 64)
    print("仿真完成。")


def _plot(times, states, sig, slow_report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[5] matplotlib 未安装，跳过出图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # 1) A 相电流（降采样包络）
    ax = axes[0]
    fs_plot = 500
    step = int(FS / fs_plot)
    ia = sig[::step, 0]
    ta = np.arange(len(ia)) * step / FS
    ax.plot(ta, ia, lw=0.6, color="#1f77b4", label="A相电流")
    ax.set_ylabel("A相电流 (pu)")
    ax.set_title("M0 端到端仿真 — 三相变频电机电流监测")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 2) 过程状态时间线
    ax = axes[1]
    y = np.array([STATE_ORDER.index(s) for s in states])
    ax.plot(times, y, drawstyle="steps-post", lw=1.2, color="#2ca02c")
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_yticklabels(STATE_ORDER)
    ax.set_ylabel("过程状态")
    ax.grid(alpha=0.3)
    # 标注注入区间
    for a, b, lbl, c in [(10, 14, "转子条边带", "#ff7f0e"), (20, 22, "堵转", "#d62728")]:
        ax.axvspan(a, b, color=c, alpha=0.15)
        ax.text((a + b) / 2, 4.4, lbl, ha="center", fontsize=9, color=c)

    # 3) 慢路径 MCSA 边带比 + 三相不平衡
    ax = axes[2]
    ts_ = [r["t"] for r in slow_report]
    ax.plot(ts_, [r["sideband_ratio"] for r in slow_report],
            marker="o", ms=3, color="#ff7f0e", label="转子条边带比")
    ax.set_ylabel("边带比")
    ax.set_xlabel("时间 (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axvspan(10, 14, color="#ff7f0e", alpha=0.15)

    os.makedirs("sim", exist_ok=True)
    out = os.path.join("sim", "m0_simulation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"[5] 已保存 {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
