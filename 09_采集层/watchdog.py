"""
毫秒级看门狗特征（M1 采集层 — 快路径）
========================================
对预处理后的每一帧（如 256 点 = 16ms @16k）计算轻量特征，供触发引擎
（trigger.py）实时消费，同时作为"看门狗"监测输入质量：

  - RMS（三相）：幅值水平 / 堵转 / 负载异常的第一手判据
  - 峰值包络 + 峰值因子：瞬态冲击 / 削波 / 电弧前兆
  - RMS 斜率：最近 N 帧平均 RMS 的最小二乘趋势 → 快速劣化检测

性能目标：~1.8ms/帧（对齐 TODO B）。纯 numpy 向量化，256×3 帧实际 <0.3ms。

与慢路径（02 特征提取）的区别：看门狗只取**触发所需的极轻特征**，
不做 FFT / MCSA（那些归慢路径，事件触发后对切片做）。

用法:
    wd = WatchdogFeatures(history=16)
    for frame in preprocessed_frames:
        snap = wd.update(frame, t_s=t)      # WatchdogSnapshot
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class WatchdogSnapshot:
    """单帧看门狗特征快照"""

    frame_idx: int                 # 帧序号（从 0 起）
    timestamp_s: float             # 帧时间（秒，调用方提供；缺省按帧间隔累计）
    rms: np.ndarray                # (C,) 三相 RMS
    envelope: np.ndarray           # (C,) 帧内峰值包络（max|·|）
    crest_factor: np.ndarray       # (C,) 峰值因子 = envelope / rms
    rms_slope: float               # 标量：平均 RMS 的滑动趋势斜率（单位 RMS/帧）
    mean_rms: float                # 平均 RMS（标量，触发引擎常用判据）
    raw: Optional[np.ndarray] = None  # (WIN, C) 原始帧（默认保留，可省内存）


def _linear_slope(values: np.ndarray) -> float:
    """最小二乘斜率（对等间隔 x=0..n-1）"""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    xm = x - x.mean()
    ym = values - values.mean()
    denom = np.dot(xm, xm)
    if denom <= 0:
        return 0.0
    return float(np.dot(xm, ym) / denom)


class WatchdogFeatures:
    """看门狗特征提取器（快路径，帧级）。"""

    def __init__(self, history: int = 16):
        if history < 2:
            raise ValueError(f"history 必须 ≥ 2，得到 {history}")
        self.history = int(history)
        self.frame_idx = 0
        self._mean_rms_hist: np.ndarray = np.zeros(0)
        self._last_t: Optional[float] = None
        self._default_t = 0.0
        self._last_snapshot: Optional[WatchdogSnapshot] = None

    def reset(self) -> None:
        """清空历史（设备重启 / 触发后重新锚定）"""
        self.frame_idx = 0
        self._mean_rms_hist = np.zeros(0)
        self._last_t = None
        self._default_t = 0.0
        self._last_snapshot = None

    @property
    def last_snapshot(self) -> Optional[WatchdogSnapshot]:
        return self._last_snapshot

    def update(
        self,
        frame: np.ndarray,
        t_s: Optional[float] = None,
    ) -> WatchdogSnapshot:
        """输入一帧预处理信号，输出看门狗特征快照。

        Args:
            frame: (WIN, C) 预处理帧（幅值保持，勿归一化 —— 快路径需绝对幅值）
            t_s: 帧绝对时间（秒）。缺省时按 256 点 @16k = 16ms 累计。

        Returns:
            WatchdogSnapshot
        """
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"帧维度不支持: {x.ndim}（期望 (WIN, C)）")

        rms = np.sqrt(np.mean(x ** 2, axis=0))
        envelope = np.max(np.abs(x), axis=0)
        crest = envelope / np.maximum(rms, 1e-12)
        mean_rms = float(np.mean(rms))

        # 时间戳：优先用调用方 t_s；缺省按 16ms 帧间隔累计
        if t_s is not None:
            ts = float(t_s)
        else:
            ts = self._default_t
            self._default_t += 0.016  # 缺省 16ms 帧间隔（256 点 @16k）

        # 平均 RMS 历史 + 斜率（快速劣化检测）
        self._mean_rms_hist = np.append(self._mean_rms_hist, mean_rms)[-self.history:]
        slope = _linear_slope(self._mean_rms_hist)

        snap = WatchdogSnapshot(
            frame_idx=self.frame_idx,
            timestamp_s=ts,
            rms=rms,
            envelope=envelope,
            crest_factor=crest,
            rms_slope=slope,
            mean_rms=mean_rms,
            raw=x,
        )
        self.frame_idx += 1
        self._last_snapshot = snap
        return snap
