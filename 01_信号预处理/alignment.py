"""
相位对齐模块

通过过零点检测实现窗口起始点的相位对齐。
对于 AC 电流分析，保证每个分析窗口从相同的相位位置（如正向过零点）开始，
能消除窗口截断位置的随机性对特征的影响。

这在电弧检测中尤为重要——电弧电流波形的"平肩部"（Shoulder）
发生在过零点附近，对齐过零点能保证该特征不被窗口切割。

数学原理:
  寻找满足以下条件的索引 k:
    signal[k-1] < 0  AND  signal[k] >= 0  (正向过零)
  (带容差 tolerance 用于处理噪声)
"""

import numpy as np
from typing import Optional, Tuple


def find_zero_crossings(
    signal: np.ndarray,
    direction: str = "positive",
    tolerance: float = 0.02,
) -> np.ndarray:
    """
    检测信号中的过零点位置

    Args:
        signal: 输入信号（应为去 DC 偏置后的信号）
        direction: "positive"(上升过零) / "negative"(下降过零) / "both"
        tolerance: 容差阈值（归一化后幅值），防止噪声误触发

    Returns:
        过零点索引数组
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if direction == "positive":
        # 寻找 signal[k-1] < -tolerance 且 signal[k] > +tolerance 的位置
        crossings = np.where(
            (signal[:-1] < -tolerance) & (signal[1:] >= tolerance)
        )[0] + 1
    elif direction == "negative":
        crossings = np.where(
            (signal[:-1] > tolerance) & (signal[1:] <= -tolerance)
        )[0] + 1
    elif direction == "both":
        crossings = np.where(
            (np.abs(signal[:-1]) > tolerance) &
            (np.sign(signal[:-1]) != np.sign(signal[1:]))
        )[0] + 1
    else:
        raise ValueError(f"未知过零方向: {direction}")

    return crossings


def align_to_zero_crossing(
    signal: np.ndarray,
    target_length: Optional[int] = None,
    direction: str = "positive",
    tolerance: float = 0.02,
) -> np.ndarray:
    """
    将信号对齐到最近的过零点

    从第一个正向过零点开始截取指定长度的窗口。

    用法:
        aligned = align_to_zero_crossing(signal, target_length=256)

    Args:
        signal: 输入信号
        target_length: 目标窗口长度，None 则使用原信号长度
        direction: 过零点方向
        tolerance: 检测容差

    Returns:
        相位对齐后的信号
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if target_length is None:
        target_length = n

    # 寻找过零点
    crossings = find_zero_crossings(signal, direction, tolerance)

    if len(crossings) == 0:
        # 没有找到过零点，返回原信号
        if n >= target_length:
            return signal[:target_length]
        else:
            return np.pad(signal, (0, target_length - n), mode="constant")

    # 选择第一个满足 target_length 且不超出边界的过零点
    for zc in crossings:
        end_idx = zc + target_length
        if end_idx <= n:
            return signal[zc:end_idx]

    # 如果没有满足长度的，使用第一个过零点并右补零
    zc = crossings[0]
    segment = signal[zc:]
    if len(segment) >= target_length:
        return segment[:target_length]
    else:
        return np.pad(segment, (0, target_length - len(segment)), mode="edge")


def extract_full_cycle(
    signal: np.ndarray,
    sample_rate: float,
    nominal_freq: float = 50.0,
    tolerance: float = 0.02,
) -> Optional[np.ndarray]:
    """
    提取一个完整的工频周期（从正向过零到下一个正向过零）

    用于稳态分析时，确保分析窗口恰好覆盖整数个工频周期，
    避免 FFT 频谱泄露。

    Args:
        signal: 输入信号
        sample_rate: 采样率 (Hz)
        nominal_freq: 标称工频 (Hz)
        tolerance: 过零容差

    Returns:
        一个完整周期的信号段，或 None（未找到）
    """
    signal = np.asarray(signal, dtype=np.float64)
    period_samples = int(sample_rate / nominal_freq)

    # 找两个连续的正向过零点
    pos_crossings = find_zero_crossings(signal, "positive", tolerance)

    if len(pos_crossings) < 2:
        return None

    # 找到最接近标准周期长度的那对过零点
    best_pair = None
    best_error = float("inf")

    for i in range(len(pos_crossings) - 1):
        cycle_len = pos_crossings[i + 1] - pos_crossings[i]
        error = abs(cycle_len - period_samples)
        if error < best_error:
            best_error = error
            best_pair = (pos_crossings[i], pos_crossings[i + 1])

    if best_pair is None:
        return None

    start, end = best_pair
    return signal[start:end + 1]


def interpolate_zero_crossing(
    signal: np.ndarray,
    idx_before: int,
    idx_after: int,
) -> float:
    """
    线性插值精确定位过零点的亚采样位置

    在离散采样中，过零点通常不在采样点上。通过线性插值
    找到更精确的过零点位置，对于高频分析（如谐波相位）非常重要。

    Args:
        signal: 信号数组
        idx_before: 过零前一个采样点索引
        idx_after: 过零后一个采样点索引

    Returns:
        精确的过零点位置（浮点索引）
    """
    y1 = signal[idx_before]
    y2 = signal[idx_after]

    # 线性插值: 在 y=0 处的 x 位置
    # x = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
    fractional = -y1 / (y2 - y1) if abs(y2 - y1) > 1e-12 else 0.0
    return float(idx_before) + fractional
