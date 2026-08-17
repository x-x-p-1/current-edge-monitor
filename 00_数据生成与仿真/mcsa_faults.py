"""
MCSA 故障合成辅助（参考 mcp-server-mcsa 的故障理论）
====================================================
按电机电流特征分析（MCSA）边带理论合成轴承 / 定子匝间短路 / 偏心 / 缺相
故障电流波形，供 `current_simulator` 管线自检与 ML 打底。

故障频率公式（参考 mcp-server-mcsa / 工业 MCSA 标准）：
  · 断条 rotor_sideband : (1 ± 2s)·fs            —— 已有（current_simulator）
  · 偏心 eccentricity   : fs ± k·fr
  · 定子匝间短路 stator : fs ± 2k·fr
  · 轴承 bearing        : fs ± k·f_defect        —— f_defect 由轴承几何计算
      f_defect ∈ {BPFO, BPFI, BSF, FTF}
  · 缺相 phase_loss     : 一相电流趋零（残留弱电流）

其中 fs = 基波频率（电网/变频输出），s = 转差率，fr = 转子机械旋转频率 (Hz)：
      fr = 2·fs·(1-s) / poles          （poles = 极数，4 极 → fr≈fs·(1-s)/2）

依赖：仅 numpy
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ============================================================
# 轴承特征频率（几何模型）
# ============================================================

def bearing_frequencies(
    fr_rot_hz: float,
    n_balls: int,
    ball_d: float,
    pitch_d: float,
    contact_angle_deg: float,
) -> Dict[str, float]:
    """由轴承几何与转子转速计算四类特征缺陷频率 (Hz)。

    Args:
        fr_rot_hz: 转子机械旋转频率 (Hz) = 2·fs·(1-s)/poles
        n_balls: 滚动体数量
        ball_d: 滚动体直径
        pitch_d: 节圆直径（滚动体中心所在圆直径）
        contact_angle_deg: 接触角 (度)

    Returns:
        dict: bpfo 外圈 / bpfi 内圈 / bsf 滚动体 / ftf 保持架 (Hz)
    """
    ca = np.radians(contact_angle_deg)
    x = ball_d / pitch_d * np.cos(ca)
    bpfo = 0.5 * n_balls * fr_rot_hz * (1.0 - x)          # 外圈缺陷
    bpfi = 0.5 * n_balls * fr_rot_hz * (1.0 + x)          # 内圈缺陷
    bsf = fr_rot_hz * (pitch_d / (2.0 * ball_d)) * (1.0 - x * x)  # 滚动体缺陷
    ftf = 0.5 * fr_rot_hz * (1.0 - x)                     # 保持架缺陷
    return {"bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def rotor_mech_freq_hz(f1: float, slip: float, poles: int) -> float:
    """转子机械旋转频率 (Hz)：fr = 2·fs·(1-s)/poles。"""
    return 2.0 * f1 * (1.0 - slip) / float(poles)


# ============================================================
# 边带注入工具
# ============================================================

def _inject_sideband_terms(
    signal: np.ndarray,
    seg: slice,
    ch: int,
    t_seg: np.ndarray,
    f_carrier: float,
    f_side: float,
    k_order: int,
    depth: float,
    amplitude: float,
) -> None:
    """在载波 f_carrier 周围注入 ±k·f_side 调制边带（MCSA 特征侧边带）。

    公式：f_carrier ± k·f_side，k=1..k_order（幅值 = depth·amplitude）。
    每相相位偏移 ch 保持三相差异化。
    """
    for kk in range(1, int(k_order) + 1):
        for sign in (+1.0, -1.0):
            fb = f_carrier + sign * kk * f_side
            signal[seg, ch] += depth * amplitude * np.sin(2.0 * np.pi * fb * t_seg + ch)


# ============================================================
# 各类故障注入
# ============================================================

def inject_eccentricity(
    signal: np.ndarray,
    seg: slice,
    t_seg: np.ndarray,
    f1: float,
    slip: float,
    poles: int,
    k_order: int,
    depth: float,
    amplitude: float,
) -> None:
    """气隙偏心：电流边带 fs ± k·fr（静态/动态偏心）。"""
    fr = rotor_mech_freq_hz(f1, slip, poles)
    for ch in range(3):
        _inject_sideband_terms(signal, seg, ch, t_seg, f1, fr, k_order, depth, amplitude)


def inject_stator_interturn(
    signal: np.ndarray,
    seg: slice,
    t_seg: np.ndarray,
    f1: float,
    slip: float,
    poles: int,
    k_order: int,
    depth: float,
    amplitude: float,
) -> None:
    """定子匝间短路：电流边带 fs ± 2k·fr（绕组不对称）。"""
    fr = rotor_mech_freq_hz(f1, slip, poles)
    for ch in range(3):
        _inject_sideband_terms(signal, seg, ch, t_seg, f1, 2.0 * fr, k_order, depth, amplitude)


def inject_bearing(
    signal: np.ndarray,
    seg: slice,
    t_seg: np.ndarray,
    f1: float,
    slip: float,
    poles: int,
    depth: float,
    amplitude: float,
    f_geometry: Dict[str, float],
    ring: str = "outer",
    k_order: int = 2,
) -> None:
    """轴承缺陷：电流边带 fs ± k·f_defect（缺陷频率经转矩波动调制进定子电流）。

    Args:
        ring: 'outer'→BPFO（外圈）/ 'inner'→BPFI（内圈）/ 'ball'→BSF / 'cage'→FTF
        f_geometry: bearing_frequencies() 返回值（含 bpfo/bpfi/bsf/ftf）
    """
    key = {"outer": "bpfo", "inner": "bpfi", "ball": "bsf", "cage": "ftf"}.get(ring, "bpfo")
    f_defect = f_geometry[key]
    for ch in range(3):
        _inject_sideband_terms(signal, seg, ch, t_seg, f1, f_defect, k_order, depth, amplitude)


def inject_phase_loss(
    signal: np.ndarray,
    seg: slice,
    depth: float,
    phase: int = 0,
    residual: float = 0.05,
) -> None:
    """缺相：指定相电流趋零（保留 residual 弱残流，模拟缺相后仍带剩磁/逆变器残流）。

    Args:
        phase: 缺相通道 0=A / 1=B / 2=C
        residual: 残留比例（默认 5%）
    """
    signal[seg, phase] *= max(float(depth), 0.0)  # depth≈0 → 完全断电；>0 → 欠相
    if depth < 0.5:
        signal[seg, phase] *= residual
