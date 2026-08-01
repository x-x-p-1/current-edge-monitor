"""
多节拍特征编排器 (v2 新增)
============================
实现"快路径 / 慢路径"分窗特征提取——项目架构的核心设计之一：

  快路径（短窗，ms 级）→ 时域 + 包络特征 → 喂触发引擎（stopping time）
  慢路径（长窗，s 级）  → 频域 + MCSA 边带 + 三相特征 → 喂诊断与 ML

物理依据：短窗 FFT 的频率分辨率 Δf = fs/N 太大，谐波 / MCSA 边带
（如 f1 ± 2·s·f1，可能仅 0.5~2Hz）在短窗下无法分辨，必须用 2~4s 长窗；
而触发 / 事件检测需要 ms 级响应，必须在短窗上做。
（v1 用单一 5.12ms 窗塞进全部特征，频域特征物理上不成立——v2 拆分解决）

输入约定：fast 用短窗（如 256 点 @16k ≈ 16ms），slow 用长窗（2~4s）。
"""

import numpy as np
from typing import Dict

from .time_domain import extract_time_domain_features
from .time_frequency import envelope_statistics
from .frequency_domain import extract_frequency_domain_features, compute_sideband_energy
from .three_phase import extract_three_phase_features


def _as_channels(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"不支持的数据维度: {x.ndim}")
    return x


def extract_fast_features(
    signal: np.ndarray,
    sample_rate: float = 16000.0,
) -> Dict[str, float]:
    """快路径特征（短窗，ms 级）：逐相时域 + 包络。

    Args:
        signal: 短窗波形，(N,) 或 (N, C)
        sample_rate: 采样率 (SPS)

    Returns:
        dict，键形如 ch0_rms / ch0_env_cv / ...
    """
    x = _as_channels(signal)
    feats: Dict[str, float] = {}
    for ch in range(x.shape[1]):
        td = extract_time_domain_features(x[:, ch], sample_rate).to_dict()
        env = envelope_statistics(x[:, ch])
        for k, v in td.items():
            feats[f"ch{ch}_{k}"] = float(v)
        for k, v in env.items():
            feats[f"ch{ch}_{k}"] = float(v)
    return feats


def extract_slow_features(
    signal: np.ndarray,
    sample_rate: float = 16000.0,
    f1: float = 50.0,
    slip: float = 0.03,
) -> Dict[str, float]:
    """慢路径特征（长窗，s 级）：逐相频域 + MCSA 边带 + 三相跨相特征。

    Args:
        signal: 长窗波形，(N,) 或 (N, C)；N 建议 ≥ 2~4s（Δf 需 < 0.5Hz）
        sample_rate: 采样率 (SPS)
        f1: 基波频率（变频器输出频率）
        slip: 假定转差率（MCSA 边带估计）

    Returns:
        dict，键形如 ch0_thd / ch0_sideband_ratio / 3p_neg_ratio / ...
    """
    x = _as_channels(signal)
    feats: Dict[str, float] = {}
    for ch in range(x.shape[1]):
        fd = extract_frequency_domain_features(
            x[:, ch], sample_rate, nominal_freq=f1
        ).to_dict()
        sb = compute_sideband_energy(x[:, ch], sample_rate, f1, slip)
        for k, v in fd.items():
            feats[f"ch{ch}_{k}"] = float(v)
        for k, v in sb.items():
            feats[f"ch{ch}_{k}"] = float(v)
    if x.shape[1] >= 3:
        feats.update(extract_three_phase_features(x, sample_rate, f1))
    return feats
