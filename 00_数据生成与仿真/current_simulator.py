"""
三相电流原始数据仿真器 (v0.1)
=============================
用途：在真实采集硬件就绪之前，按《采集侧确认清单》的原始数据契约生成合成
      三相电流原始数据，用于：
        1) 测试预处理/特征/触发/切片全管线（管线自检）
        2) 作为真实采集数据的对照体检基准（SNR/同步性/连续性）
        3) 生成带标签的合成故障样本，为 ML 路线（正常建模/异常检测/分类）打底

参考：
  - 西门子 LGF 波形发生算法族（Sinus/SawTooth/Rectangle）的边缘管线自检理念
  - 三菱 MCSA 边带模型（转子条故障边带位于 f1 ± 2·s·f1）

依赖：仅 numpy

用法示例：
    from current_simulator import generate_dataset, Fault
    sig, ts, meta = generate_dataset(
        fs=16000, duration=10.0, f1=50.0,
        faults=[Fault(kind="rotor_sideband", start=5.0, dur=3.0, slip=0.03, depth=0.02)],
    )
    np.save("sim/abc_normal.npy", sig)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 故障 / 异常定义
# ============================================================

@dataclass
class Fault:
    """合成故障/异常注入定义。

    kind:
      stall          堵转：电流幅值突增（×depth）
      load_step      负载突变：幅值阶跃（×depth）
      rotor_sideband MCSA 转子条边带：注入 f1 ± 2·k·s·f1（k=1..k 阶，强度 depth）
      spike          随机冲击脉冲（接触不良/毛刺），density 控制密度
      harmonic       谐波畸变升高：注入高次谐波（强度 depth）
      unbalance      三相不平衡：B 相降 C 相升
      degradation    全程缓变劣化：幅值线性升至 1+depth（慢性故障）
      arc            电弧：零休（平肩）+ 高频随机爆发（强度 depth）
    """
    kind: str = "stall"
    start: float = 0.0          # 起始时间 (s)
    dur: float = 1.0            # 持续时间 (s)；degradation 全程生效
    depth: float = 0.5          # 幅值倍率 / 强度
    slip: float = 0.03          # (rotor_sideband) 转差率
    k: int = 2                  # (rotor_sideband) 边带阶次上限
    params: dict = field(default_factory=dict)


# ============================================================
# 内部工具
# ============================================================

def _make_t(fs: float, duration: float) -> np.ndarray:
    return np.arange(int(duration * fs)) / fs


def _quantize(
    x: np.ndarray,
    adc_bits: int,
    full_scale: float,
    return_clip: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """模拟 ADC 量化与削波。"""
    levels = float(2 ** (adc_bits - 1))
    q = np.clip(x, -full_scale, full_scale)
    q = np.round(q / (full_scale / levels)) * (full_scale / levels)
    if return_clip:
        clipped = np.abs(x) > full_scale
        return q, clipped
    return q, None


def _apply_fault(
    signal: np.ndarray,
    t: np.ndarray,
    f: Fault,
    fs: float,
    f1: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(t)
    if f.kind == "degradation":
        ramp = np.linspace(1.0, 1.0 + f.depth, n)
        return signal * ramp[:, None]

    i0 = max(0, int(f.start * fs))
    i1 = min(n, int((f.start + f.dur) * fs))
    if i1 <= i0:
        return signal
    seg = slice(i0, i1)

    if f.kind in ("stall", "load_step"):
        signal[seg] *= f.depth

    elif f.kind == "rotor_sideband":
        for ch in range(3):
            for kk in range(1, f.k + 1):
                for sign in (+1, -1):
                    fb = f1 + sign * 2 * kk * f.slip * f1
                    signal[seg, ch] += (
                        f.depth * amplitude * np.sin(2 * np.pi * fb * t[seg] + ch)
                    )

    elif f.kind == "spike":
        density = f.params.get("density", 0.001)
        nsp = max(1, int((f.dur * fs) * density))
        idx = rng.integers(i0, i1, nsp)
        for ch in range(3):
            signal[idx, ch] += f.depth * amplitude * rng.choice([-1.0, 1.0], nsp)

    elif f.kind == "harmonic":
        for ch in range(3):
            for kk in (5, 7, 11, 13, 17, 19):
                signal[seg, ch] += (
                    f.depth * amplitude * np.sin(2 * np.pi * kk * f1 * t[seg] + ch)
                )

    elif f.kind == "unbalance":
        signal[seg, 1] *= (1.0 - f.depth)
        signal[seg, 2] *= (1.0 + f.depth)

    elif f.kind == "arc":
        for ch in range(3):
            seg_vals = signal[seg, ch]
            # 零休：过零附近钳位到 0（模拟电弧平肩）
            zero_mask = np.abs(seg_vals) < 0.02 * amplitude
            seg_vals[zero_mask] = 0.0
            # 高频随机爆发（强度随幅值衰减）
            burst = rng.normal(0.0, f.depth * amplitude, seg_vals.shape)
            burst[np.abs(seg_vals) > 0.05 * amplitude] *= 0.1
            seg_vals += burst
            signal[seg, ch] = seg_vals

    else:
        raise ValueError(f"未知故障类型: {f.kind}")

    return signal


# ============================================================
# 主生成接口
# ============================================================

def generate_dataset(
    fs: float = 16000.0,
    duration: float = 10.0,
    f1: float = 50.0,
    amplitude: float = 1.0,
    harmonics: Optional[Dict[int, float]] = None,
    fsw: float = 8000.0,
    pwm_depth: float = 0.03,
    snr_db: float = 60.0,
    adc_bits: int = 24,
    full_scale: float = 2.0,
    imbalance: float = 0.0,
    load_profile: Optional[List[Tuple[float, float, float]]] = None,
    faults: Optional[List[Fault]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """生成三相电流原始数据。

    Args:
        fs: 采样率 (SPS/通道)
        duration: 时长 (s)
        f1: 基波频率 (Hz，变频器输出频率)
        amplitude: 基波幅值（标幺或安培）
        harmonics: {谐波次数: 相对基波幅值}
        fsw: 变频器开关频率 (Hz)
        pwm_depth: PWM 纹波相对幅值
        snr_db: 信噪比 (dB)，inf 表示无噪声
        adc_bits: 量化位深
        full_scale: 量化满量程
        imbalance: 三相不平衡度（B 相 +imbalance，C 相 -imbalance）
        load_profile: [(t0, t1, amp), ...] 负载包络（过程状态）
        faults: Fault 列表
        seed: 随机种子

    Returns:
        (signal, timestamps_ns, metadata)
          signal: (N, 3) float —— 三相电流（标幺/安培）
          timestamps_ns: (N,) int64 —— 模拟 UTC 时戳（步长 1e9/fs ns）
          metadata: dict —— 契约参数 + 量化/削波统计
    """
    if harmonics is None:
        harmonics = {5: 0.12, 7: 0.06, 11: 0.03, 13: 0.02}  # 电机特征谐波
    if faults is None:
        faults = []

    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = _make_t(fs, duration)

    # 负载包络（过程状态：启动/空载/负载）
    if load_profile is None:
        env = np.ones(n)
    else:
        env = np.zeros(n)
        for t0, t1, a in load_profile:
            i0, i1 = int(t0 * fs), int(t1 * fs)
            env[i0:i1] = a
        env[env == 0] = 1.0

    phase_shift = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])  # A/B/C
    imb = np.array([1.0, 1.0 + imbalance, 1.0 - imbalance])

    signal = np.zeros((n, 3))
    for ch, ph in enumerate(phase_shift):
        s = amplitude * env * np.sin(2.0 * np.pi * f1 * t + ph)
        for k, a in harmonics.items():
            s += amplitude * env * a * np.sin(2.0 * np.pi * k * f1 * t + k * ph + 0.5)
        signal[:, ch] = s * imb[ch]

    # PWM 开关纹波
    for ch in range(3):
        signal[:, ch] += pwm_depth * amplitude * np.sin(2.0 * np.pi * fsw * t + ch)

    # 高斯噪声
    if snr_db < np.inf:
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_std = signal_rms / (10.0 ** (snr_db / 20.0))
        signal += rng.normal(0.0, noise_std, signal.shape)

    # 施加故障
    for f in faults:
        signal = _apply_fault(signal, t, f, fs, f1, amplitude, rng)

    # 时戳（模拟 UTC）
    timestamps_ns = np.arange(n, dtype=np.int64) * int(1e9 / fs)

    # 量化 + 削波统计
    signal_q, clip = _quantize(signal, adc_bits, full_scale, return_clip=True)

    meta = {
        "fs": fs,
        "duration": duration,
        "f1": f1,
        "adc_bits": adc_bits,
        "full_scale": full_scale,
        "snr_db": snr_db,
        "channels": 3,
        "sync": True,
        "clip_ratio": float(clip.mean()),
        "units": "A (per-unit)",
        "faults": [f.kind for f in faults],
    }
    return signal_q, timestamps_ns, meta


# ============================================================
# 演示 / 自检
# ============================================================

def _demo() -> None:
    """生成一份正常 + 一份含多故障的样本，打印统计并落盘。"""
    import os

    os.makedirs("sim", exist_ok=True)

    # 正常样本（带负载周期）
    sig, ts, meta = generate_dataset(
        duration=10.0,
        f1=50.0,
        load_profile=[(0.0, 2.0, 0.3), (2.0, 8.0, 1.0), (8.0, 10.0, 0.3)],
    )
    np.save("sim/abc_normal.npy", sig)
    print(f"[正常] {sig.shape}, Fs={meta['fs']}, 削波={meta['clip_ratio']:.4f}")

    # 多故障样本（ML 打底用）
    sig2, ts2, meta2 = generate_dataset(
        duration=20.0,
        f1=50.0,
        faults=[
            Fault(kind="load_step", start=3.0, dur=0.5, depth=1.6),
            Fault(kind="rotor_sideband", start=6.0, dur=4.0, slip=0.03, depth=0.02),
            Fault(kind="spike", start=11.0, dur=2.0, depth=0.5, params={"density": 0.002}),
            Fault(kind="stall", start=15.0, dur=1.0, depth=1.8),
        ],
    )
    np.save("sim/abc_faults.npy", sig2)
    print(f"[多故障] {sig2.shape}, 故障={meta2['faults']}")

    print("已保存 sim/abc_normal.npy, sim/abc_faults.npy")


if __name__ == "__main__":
    _demo()
