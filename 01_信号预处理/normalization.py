"""
信号归一化模块

将滤波后的信号幅值统一到相同的尺度范围，
消除不同负载电流幅值差异对模型的影响。

类比西门子 DKW 的无量纲归一化思想:
  通过归一化使得模型关注的是波形形态（Shape）而非绝对幅值（Amplitude），
  从而实现对不同额定电流负载的通用检测能力。
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class NormalizationMethod(str, Enum):
    """归一化方法枚举"""
    ZSCORE = "zscore"     # Z-Score: (x - μ) / σ，结果 ~ N(0, 1)
    MINMAX = "minmax"     # Min-Max: (x - min) / (max - min)，结果 ∈ [0, 1]
    RMS = "rms"           # RMS 归一化: x / I_rms
    PEAK = "peak"         # 峰值归一化: x / max(|x|)
    NONE = "none"         # 不归一化


def normalize_signal(
    data: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.ZSCORE,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    将输入信号归一化

    Args:
        data: 输入信号，shape (N,) 或 (batch, N)
        method: 归一化方法
        eps: 防止除零的小常数

    Returns:
        归一化后的信号
    """
    if method == NormalizationMethod.NONE:
        return data

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        return _normalize_1d(data, method, eps)
    elif data.ndim == 2:
        # (N, C) 时间×通道：逐通道沿时间轴归一化
        return np.column_stack(
            [_normalize_1d(data[:, ch], method, eps) for ch in range(data.shape[1])]
        )
    else:
        raise ValueError(f"不支持的数据维度: {data.ndim}")


def _normalize_1d(
    signal: np.ndarray,
    method: NormalizationMethod,
    eps: float,
) -> np.ndarray:
    """单通道归一化"""
    if method == NormalizationMethod.ZSCORE:
        mu = np.mean(signal)
        sigma = np.std(signal)
        if sigma < eps:
            return signal - mu  # 常数信号，仅去均值
        return (signal - mu) / sigma

    elif method == NormalizationMethod.MINMAX:
        s_min = np.min(signal)
        s_max = np.max(signal)
        denom = s_max - s_min
        if denom < eps:
            return np.zeros_like(signal)
        return (signal - s_min) / denom

    elif method == NormalizationMethod.RMS:
        rms = np.sqrt(np.mean(signal ** 2))
        if rms < eps:
            return signal
        return signal / rms

    elif method == NormalizationMethod.PEAK:
        peak = np.max(np.abs(signal))
        if peak < eps:
            return signal
        return signal / peak

    else:
        raise ValueError(f"未知归一化方法: {method}")


def compute_normalization_params(
    signal: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.ZSCORE,
) -> dict:
    """
    计算归一化参数（用于训练/推理一致性）

    在训练集上计算参数，保存后在推理时复用，
    确保归一化一致性。

    Returns:
        参数字典，如 {"mu": 0.1, "sigma": 0.5}
    """
    signal = np.asarray(signal, dtype=np.float64)

    if method == NormalizationMethod.ZSCORE:
        return {"mu": float(np.mean(signal)), "sigma": float(np.std(signal))}
    elif method == NormalizationMethod.MINMAX:
        return {"min": float(np.min(signal)), "max": float(np.max(signal))}
    elif method == NormalizationMethod.RMS:
        return {"rms": float(np.sqrt(np.mean(signal ** 2)))}
    elif method == NormalizationMethod.PEAK:
        return {"peak": float(np.max(np.abs(signal)))}
    else:
        return {}


def apply_normalization_params(
    signal: np.ndarray,
    params: dict,
    method: NormalizationMethod = NormalizationMethod.ZSCORE,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    使用预计算的参数进行归一化（推理时使用）

    Args:
        signal: 输入信号
        params: compute_normalization_params 返回的参数字典
        method: 归一化方法
        eps: 防止除零

    Returns:
        归一化后的信号
    """
    signal = np.asarray(signal, dtype=np.float64)

    if method == NormalizationMethod.ZSCORE:
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        if abs(sigma) < eps:
            return signal - mu
        return (signal - mu) / sigma

    elif method == NormalizationMethod.MINMAX:
        s_min = params.get("min", 0.0)
        s_max = params.get("max", 1.0)
        denom = s_max - s_min
        if denom < eps:
            return np.zeros_like(signal)
        return (signal - s_min) / denom

    elif method == NormalizationMethod.RMS:
        rms = params.get("rms", 1.0)
        if rms < eps:
            return signal
        return signal / rms

    elif method == NormalizationMethod.PEAK:
        peak = params.get("peak", 1.0)
        if peak < eps:
            return signal
        return signal / peak

    else:
        return signal
