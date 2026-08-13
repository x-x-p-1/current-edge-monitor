"""
全速率环形缓冲（M1 采集层 — P0 主线）
======================================
固定容量、常开、O(1) 写入的环形缓冲，用于"全速率常开采集"：
原始采样持续写入，异常触发时从缓冲中冻结**预触发**波形，
配合后触发补录形成完整切片（见 slice_capture.py）。

数据布局: (N, C) 时间×通道，与 01_信号预处理 的契约一致。

读写语义:
  - write(samples): O(1) 摊销追加，溢出覆盖最旧样本
  - get_last(n): 最近 n 个样本（按时间顺序）
  - slice_range(start, stop): 按**绝对索引**范围取样本，
    越界部分裁剪（预触发早于缓冲起点时，可据此判断切片是否完整）

性能: 写入为 memcpy 级别；读取仅在切片/诊断时发生（低频），O(n)。
"""
from typing import Optional

import numpy as np


class RingBuffer:
    """固定容量环形缓冲（时间×通道，溢出覆盖最旧）。

    Args:
        capacity: 容量（采样点数，每通道）。超过后最旧样本被覆盖。
        channels: 通道数（三相 = 3）
    """

    def __init__(self, capacity: int, channels: int = 3):
        if capacity <= 0:
            raise ValueError(f"capacity 必须 > 0，得到 {capacity}")
        if channels <= 0:
            raise ValueError(f"channels 必须 > 0，得到 {channels}")
        self.capacity = int(capacity)
        self.channels = int(channels)
        self._buf = np.zeros((self.capacity, self.channels), dtype=np.float64)
        self._count = 0       # 历史累计写入样本数（含被覆盖的）
        self._overflow = 0    # 被覆盖的样本数
        # 不变式：绝对索引 i 的样本存储于 _buf[i % capacity]

    # ────────────────────────────
    # 状态
    # ────────────────────────────
    @property
    def n_samples(self) -> int:
        """当前可读样本数（≤ capacity）"""
        return min(self._count, self.capacity)

    @property
    def filled(self) -> bool:
        """缓冲是否已写满过"""
        return self._count >= self.capacity

    @property
    def overflow_samples(self) -> int:
        """已因容量限制被覆盖掉的样本数"""
        return self._overflow

    def first_available_index(self) -> int:
        """第一个仍可读样本的绝对索引（0-based，全局时间轴）"""
        return max(0, self._count - self.capacity)

    def reset(self) -> None:
        """清空缓冲（设备重启 / 重新锚定）"""
        self._buf.fill(0.0)
        self._count = 0
        self._overflow = 0

    # ────────────────────────────
    # 写入
    # ────────────────────────────
    def write(self, samples: np.ndarray) -> None:
        """追加采样。samples: (M,) 或 (M, C)。

        溢出时覆盖最旧样本；写入 O(M)（memcpy 级）。
        """
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] != self.channels:
            got = x.shape[1] if x.ndim == 2 else 1
            raise ValueError(f"通道数不匹配: 期望 {self.channels}，得到 {got}")
        m = x.shape[0]
        if m <= 0:
            return

        # 统一映射：绝对索引 i → 位置 i % capacity（fancy 赋值保持环形覆盖语义）
        pos = (self._count + np.arange(m)) % self.capacity
        self._buf[pos] = x

        self._count += m
        self._overflow = max(self._overflow, self._count - self.capacity)

    # ────────────────────────────
    # 读取（低频：切片 / 诊断）
    # ────────────────────────────
    def _pos(self, idx: int) -> int:
        """绝对索引 → 环形物理位置（第 idx 个写入样本存于 idx % capacity）"""
        return idx % self.capacity

    def slice_range(self, start: int, stop: int) -> np.ndarray:
        """按绝对索引范围取样本 [start, stop)（按时间顺序）。

        越界部分被裁剪：
          - start < 缓冲起点 → 预触发不完整（可结合 first_available_index 判断）
          - stop > 已写入总数 → 尚未采集到
        完全越界返回空 (0, C)。

        Args:
            start: 起始绝对索引（含）
            stop: 结束绝对索引（不含）

        Returns:
            (N, C) 样本，时间顺序
        """
        s = max(start, self.first_available_index())
        e = min(stop, self._count)
        if e <= s:
            return np.empty((0, self.channels))
        idx = np.arange(s, e)
        return self._buf[idx % self.capacity]

    def get_last(self, n: int) -> np.ndarray:
        """最近 n 个样本（时间顺序；不足 n 时返回全部可读）"""
        if n <= 0:
            return np.empty((0, self.channels))
        return self.slice_range(self._count - n, self._count)

    def as_array(self) -> np.ndarray:
        """全部可读样本（时间顺序）"""
        return self.slice_range(self.first_available_index(), self._count)

    def __len__(self) -> int:
        return self.n_samples
