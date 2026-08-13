"""
数值守卫（10 鲁棒性 — R1/R2）
==============================
全链路数值稳健：任何数值出口都必须经过 isfinite 守卫，
NaN/Inf 不得进入特征 / 状态机 / 报警；除零用安全除法替代。

对齐《08_鲁棒性/鲁棒性清单.md》：
  R1 NaN/Inf 传播 → 注入后不崩溃、标记无效帧并跳过、恢复后自动复位
  R2 除零/病态计算 → 除零守卫 + epsilon，低于阈值输出"无效"而非 0 或 ±Inf
"""
import math
from typing import Tuple

import numpy as np


def frame_is_valid(x: np.ndarray) -> bool:
    """帧是否全有限（无 NaN / Inf）"""
    return bool(np.all(np.isfinite(np.asarray(x, dtype=np.float64))))


def sanitize_nan_inf(x: np.ndarray, fill: float = 0.0) -> Tuple[np.ndarray, bool]:
    """清洗 NaN/Inf，返回 (清洗后数据, 是否有非有限值)。

    有非有限值 → 返回 True，调用方应据此标记"无效帧/无效段"；
    清洗后用 fill 替换，避免下游除零 / 滤波发散。
    """
    a = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(a)
    if finite.all():
        return a, False
    out = a.copy()
    out[~finite] = fill
    return out, True


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = 1e-12,
    default: float = 0.0,
) -> np.ndarray:
    """除零守卫：denominator 绝对值 < eps 时输出 default，否则正常除。

    Args:
        numerator / denominator: 同形状数组
        eps: 除零判定阈值
        default: 无效除法输出（避免 ±Inf / NaN 进入下游）
    """
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast_shapes(n.shape, d.shape), default, dtype=np.float64)
    mask = np.abs(d) >= eps
    out[mask] = n[mask] / d[mask]
    return out


def safe_log10(x: np.ndarray, floor: float = 1e-12, default: float = -120.0) -> np.ndarray:
    """对数守卫：x ≤ floor 时输出 default（防止 log(0) / log(负) → NaN）"""
    a = np.asarray(x, dtype=np.float64)
    out = np.full_like(a, default)
    mask = a > floor
    out[mask] = np.log10(a[mask])
    return out


def scale_clip(x: np.ndarray, lo: float = -1e30, hi: float = 1e30) -> np.ndarray:
    """幅值钳制（高增益滤波 / 异常样本防御），防止 ±Inf 或超大值进入下游"""
    return np.clip(np.asarray(x, dtype=np.float64), lo, hi)


class NumericGuard:
    """每帧数值守卫（有状态：记录连续无效帧，供降级判断）。"""

    def __init__(self, max_invalid_streak: int = 5):
        self.max_invalid_streak = int(max_invalid_streak)
        self.invalid_streak = 0
        self.total_invalid = 0

    def reset(self) -> None:
        self.invalid_streak = 0
        self.total_invalid = 0

    def check(self, x: np.ndarray, fill: float = 0.0) -> Tuple[np.ndarray, bool]:
        """检查并清洗一帧。返回 (清洗后帧, 是否有效)。

        无效（含 NaN/Inf）→ invalid_streak+1；连续超阈值视为输入劣化。
        """
        cleaned, has_nonfinite = sanitize_nan_inf(x, fill=fill)
        if has_nonfinite:
            self.invalid_streak += 1
            self.total_invalid += 1
        else:
            self.invalid_streak = 0
        return cleaned, not has_nonfinite

    @property
    def degraded(self) -> bool:
        """是否处于降级态（连续无效帧超阈值）"""
        return self.invalid_streak >= self.max_invalid_streak
