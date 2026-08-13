"""
混沌注入器（10 鲁棒性 — 4.2 fault injection）
==============================================
向信号流注入各类故障，验证"能犯病、能识别、能恢复"：
进程存活 100%、每类注入有明确"标记 + 降级 + 恢复"路径。

对齐《08_鲁棒性/鲁棒性清单.md》§4.2：
  注入类型：NaN / ±Inf / 断线(常值) / 直流偏置 / ADC 削波 / 丢样 /
           相位错序 / 突发异常
  注入时机：快路径任一帧、慢路径任一窗、事件后、重启后

用法（与 AcquisitionEngine / 测试配合）:
    chaos = ChaosInjector(seed=0, full_scale=3.3)
    data, kind = chaos.inject(normal_chunk, kind="nan")
    saved = engine.feed(data, t_s=t)   # 应不崩溃、被标记、后续恢复
"""
from typing import Optional, Tuple

import numpy as np


class ChaosInjector:
    """混沌注入器：对采样块注入单类故障，返回 (注入后数据, 故障类型)。"""

    # 数据类注入（返回数组）
    DATA_KINDS = (
        "nan", "inf", "dropout", "dc_bias",
        "clip", "sample_drop", "phase_swap",
    )
    # 全种类（含突发异常：返回 None，触发引擎异常路径）
    ALL_KINDS = DATA_KINDS + ("exception",)

    def __init__(
        self,
        seed: int = 0,
        full_scale: float = 1.0,
        inject_ratio: float = 0.3,
    ):
        self.rng = np.random.default_rng(seed)
        self.full_scale = float(full_scale)
        self.inject_ratio = float(inject_ratio)   # 注入点占比（缺省/采样）

    def inject(
        self,
        chunk: np.ndarray,
        kind: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, str]:
        """对采样块注入故障。

        Args:
            chunk: (N,) 或 (N, C)
            kind: 故障类型；None 时随机挑一个数据类注入
            **kwargs: 覆盖参数（如 inject_ratio / full_scale）

        Returns:
            (data, kind) —— 数据类注入返回数组；
            kind=="exception" 返回 (None, "exception")（模拟突发异常）
        """
        if kind is None:
            kind = self.rng.choice(self.DATA_KINDS)
        if kind not in self.ALL_KINDS:
            raise ValueError(f"未知注入类型 {kind}，可用: {self.ALL_KINDS}")

        x = np.asarray(chunk, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        ratio = float(kwargs.get("inject_ratio", self.inject_ratio))
        fs = float(kwargs.get("full_scale", self.full_scale))

        if kind == "nan":
            out = x.copy()
            mask = self.rng.random(x.shape) < ratio
            out[mask] = np.nan
        elif kind == "inf":
            out = x.copy()
            mask = self.rng.random(x.shape) < ratio
            out[mask] = np.sign(self.rng.standard_normal(x.shape)[mask]) * np.inf
        elif kind == "dropout":
            # 传感器断线：整段置常值（保留首样本）
            out = np.full_like(x, x[0] if len(x) else 0.0)
        elif kind == "dc_bias":
            bias = float(kwargs.get("bias", 0.5 * fs))
            out = x + bias
        elif kind == "clip":
            # ADC 削波：超满量程部分截平
            out = np.clip(x, -fs, fs)
        elif kind == "sample_drop":
            # 丢样：返回空块（时戳跳跃由调用方模拟）
            out = np.empty((0, x.shape[1]))
        elif kind == "phase_swap":
            # 相位错序：交换 B/C 相（A-C-B）
            out = x.copy()
            if x.shape[1] >= 3:
                out[:, [1, 2]] = x[:, [2, 1]]
        elif kind == "exception":
            return None, kind
        else:  # pragma: no cover
            raise AssertionError(kind)

        return (out[:, 0] if single else out), kind
