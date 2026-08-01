"""
三相特征提取模块 (v2 新增)
==========================
面向三相变频电机的跨相特征（输入约定 (N, 3) 时间×通道，A/B/C，建议已去 DC）：
  - 三相幅值平衡度 / 不平衡度（GB/T 15543 思路）
  - 基频相角偏差（理想 120° 校验）
  - 对称分量（正序 / 负序 / 零序，Fortescue 变换）
  - 三相皮尔逊相关矩阵

对应算法清单：三菱 #8 皮尔逊相关系数（跨相对齐）；三相不平衡（GB/T 15543-2008）。
"""

import numpy as np
from typing import Dict


def _as_3ph(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("三相特征需要 (N, 3) 输入")
    return x


def phase_rms(x: np.ndarray) -> np.ndarray:
    """三相每相 RMS → (3,)"""
    x = _as_3ph(x)
    return np.sqrt(np.mean(x ** 2, axis=0))


def phase_balance(x: np.ndarray) -> Dict[str, float]:
    """三相幅值平衡度：balance_ratio（min/max）与 unbalance_pct（GB/T 15543 式）"""
    rms = phase_rms(x)
    r_max, r_min = float(rms.max()), float(rms.min())
    if (r_max + r_min) < 1e-12:
        return {"balance_ratio": 1.0, "unbalance_pct": 0.0}
    unbalance_pct = 200.0 * (r_max - r_min) / (r_max + r_min)
    return {"balance_ratio": float(r_min / r_max), "unbalance_pct": float(unbalance_pct)}


def fundamental_phasors(x: np.ndarray, sample_rate: float, f1: float) -> np.ndarray:
    """基频处三相复相量（单 bin DFT）→ (3,) complex"""
    x = _as_3ph(x)
    N = len(x)
    k = int(round(f1 * N / sample_rate))
    if k <= 0 or k >= N // 2:
        k = max(1, min(N // 2 - 1, k))
    n = np.arange(N)
    w = np.exp(-2j * np.pi * k * n / N)
    return np.array([np.sum(x[:, ch] * w) * 2.0 / N for ch in range(3)])


def phase_angle_errors(x: np.ndarray, sample_rate: float, f1: float) -> Dict[str, float]:
    """基频相角偏差（理想 120°），返回各相间偏差(rad)与最大偏差"""
    V = fundamental_phasors(x, sample_rate, f1)
    ang = np.angle(V)
    ideal = 2.0 * np.pi / 3.0

    def err(d: float) -> float:
        return float(abs(((d - ideal + np.pi) % (2.0 * np.pi)) - np.pi))

    d_ab = (ang[0] - ang[1]) % (2.0 * np.pi)
    d_bc = (ang[1] - ang[2]) % (2.0 * np.pi)
    d_ca = (ang[2] - ang[0]) % (2.0 * np.pi)
    es = [err(d_ab), err(d_bc), err(d_ca)]
    return {"phase_err_ab": es[0], "phase_err_bc": es[1],
            "phase_err_ca": es[2], "phase_err_max": max(es)}


def symmetry_components(x: np.ndarray, sample_rate: float, f1: float) -> Dict[str, float]:
    """对称分量（正序 V1 / 负序 V2 / 零序 V0）与相对比例"""
    V = fundamental_phasors(x, sample_rate, f1)
    a = np.exp(2j * np.pi / 3.0)
    v1 = (V[0] + a * V[1] + a ** 2 * V[2]) / 3.0
    v2 = (V[0] + a ** 2 * V[1] + a * V[2]) / 3.0
    v0 = (V[0] + V[1] + V[2]) / 3.0
    m1 = abs(v1)
    return {
        "v1_pos": float(m1),
        "v2_neg": float(abs(v2)),
        "v0_zero": float(abs(v0)),
        "neg_ratio": float(abs(v2) / m1) if m1 > 1e-12 else 0.0,
        "zero_ratio": float(abs(v0) / m1) if m1 > 1e-12 else 0.0,
    }


def phase_correlation(x: np.ndarray) -> Dict[str, float]:
    """三相皮尔逊相关（三菱 #8 思路，跨相）"""
    x = _as_3ph(x)
    c = np.corrcoef(x, rowvar=False)
    return {"corr_ab": float(c[0, 1]), "corr_bc": float(c[1, 2]), "corr_ca": float(c[2, 0])}


def extract_three_phase_features(x: np.ndarray, sample_rate: float, f1: float) -> Dict[str, float]:
    """汇总三相特征 → dict（键前缀 3p_）"""
    feats: Dict[str, float] = {}
    feats.update({f"3p_{k}": v for k, v in phase_balance(x).items()})
    feats.update({f"3p_{k}": v for k, v in phase_angle_errors(x, sample_rate, f1).items()})
    feats.update({f"3p_{k}": v for k, v in symmetry_components(x, sample_rate, f1).items()})
    feats.update({f"3p_{k}": v for k, v in phase_correlation(x).items()})
    return feats
