"""
电流-电压交叉验证特征 (v0.3 新增)
==================================
原理：电流和电压通过电机阻抗 Z=V/I、功率因数 cosφ 物理耦合。健康电机两者关系
稳定可预测；**同时跟踪两者**可区分根因：

  · 电机/负载侧问题（堵转 / 绕组短路 / 负载突变）→ 电流先变，电压正常，Z 变化
  · 供电侧问题（电压跌落 / 电压不平衡 / 缺相）   → 电压先变，电流跟随，Z 基本不变
  · 电流不平衡 与 电压不平衡 的比例异常 → 电机侧不对称 vs 供电侧

依赖：仅 numpy。输入约定：v/i 均为 (N,) 单相或 (N,C) 三相，时间×通道。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def estimate_impedance(v: np.ndarray, i: np.ndarray) -> float:
    """每相阻抗 Z = V_rms / I_rms (Ω)。

    堵转/绕组短路：I 激增 → Z 骤降；连接/接触不良：I 受限 → Z 升高。
    """
    v_rms = float(np.sqrt(np.mean(v ** 2)))
    i_rms = float(np.sqrt(np.mean(i ** 2)))
    if i_rms < 1e-9:
        return float("inf")
    return v_rms / i_rms


def estimate_power_factor(v: np.ndarray, i: np.ndarray, fs: float, f1: float) -> float:
    """功率因数 cosφ：用基波 DFT 求 V 与 I 的相位差，再取余弦。

    物理：空载 PF 低（励磁无功），满载 PF≈额定；劣化/去磁会使 PF 异常。
    """
    v = np.asarray(v, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    n = min(len(v), len(i))
    t = np.arange(n) / fs
    ref = np.exp(-1j * 2.0 * np.pi * f1 * t)
    V = 2.0 * np.dot(v[:n], ref) / n
    I = 2.0 * np.dot(i[:n], ref) / n
    if abs(V) < 1e-12 or abs(I) < 1e-12:
        return 0.0
    ang = np.angle(V) - np.angle(I)          # V 超前 I → 正（感性负载）
    return float(np.clip(np.cos(ang), -1.0, 1.0))


def unbalance_factor(x: np.ndarray) -> float:
    """三相不平衡度（NEMA 式）：(max-min)/max ×100%。x 为 (N,3)。"""
    x = np.asarray(x)
    if x.ndim == 1:
        return 0.0
    rms = np.sqrt(np.mean(x ** 2, axis=0))
    if rms.max() < 1e-9:
        return 0.0
    return float((rms.max() - rms.min()) / rms.max() * 100.0)


def cross_report(
    v: np.ndarray,
    i: np.ndarray,
    fs: float,
    f1: float,
) -> Dict[str, float]:
    """电流-电压交叉验证指标包。

    Args:
        v: 三相电压 (N,3) 或单相 (N,)
        i: 三相电流 (N,3) 或单相 (N,)
        fs: 采样率
        f1: 基波频率

    Returns:
        {v_rms, i_rms, z_ohm, pf, v_unbalance_pct, i_unbalance_pct}
    """
    v = np.asarray(v, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    if i.ndim == 1:
        i = i[:, None]

    ch = 0  # 用 A 相估计 Z / PF（三相平衡时一致）
    return {
        "v_rms": float(np.sqrt(np.mean(v[:, ch] ** 2))),
        "i_rms": float(np.sqrt(np.mean(i[:, ch] ** 2))),
        "z_ohm": estimate_impedance(v[:, ch], i[:, ch]),
        "pf": estimate_power_factor(v[:, ch], i[:, ch], fs, f1),
        "v_unbalance_pct": unbalance_factor(v),
        "i_unbalance_pct": unbalance_factor(i),
    }


def classify_side(r: Dict[str, float], ref: Optional[Dict[str, float]] = None) -> str:
    """供电侧 vs 电机侧 判别（基于与参考健康值的相对变化）。

    规则：
      · Z 相对健康值下降 >30% 且 V 基本不变 → 电机/负载侧（堵转/绕组短路）
      · V 下降 >5% 且 |ΔZ|/Z 相对 <20% → 供电侧（电压跌落）
      · 电流不平衡 >> 电压不平衡（>3x）→ 电机侧不对称
      · 电压不平衡明显（>2%）→ 供电侧不平衡
    """
    if ref is None:
        return "正常（无参考基线）"

    d_z = (r["z_ohm"] - ref["z_ohm"]) / max(ref["z_ohm"], 1e-9)
    d_v = (r["v_rms"] - ref["v_rms"]) / max(ref["v_rms"], 1e-9)

    if r["v_unbalance_pct"] > 2.0:
        return "供电侧：电压不平衡显著"
    if d_v < -0.05 and abs(d_z) < 0.20:
        return "供电侧：电压跌落（V 跌而 Z 稳定）"
    if d_z < -0.30 and abs(d_v) < 0.05:
        return "电机/负载侧：阻抗骤降（堵转/绕组短路）"
    if r["i_unbalance_pct"] > 3.0 * max(r["v_unbalance_pct"], 0.5):
        return "电机侧：电流不平衡 >> 电压不平衡（电机不对称）"
    return "正常（V/I 关系稳定）"
