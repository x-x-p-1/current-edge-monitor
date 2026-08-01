"""
信号预处理 — 主流水线模块 (v2 — 三相/VFD 方向重构)
====================================================
将原始 ADC 采样数据（单相 (N,) 或三相 (N, C)）经过预处理，
输出干净、对齐、归一化的信号，供特征提取与模型推理使用。

预处理管线:
    原始采样 → 直流偏置去除 → 带通滤波 → 归一化 → [相位对齐] → 干净信号

v2 设计变更（对齐三相变频电机方向）:
  1. 通道结构：支持 (N, C) 三相，所有步骤逐通道执行；单通道 (N,) 完全兼容。
  2. 带通默认参数改为 VFD 友好（lowcut=5Hz / highcut=4000Hz），
     修掉 v1 中"带通 2500Hz 滤掉高频诊断特征"的自相矛盾。
  3. 新增流式处理（process_streaming）：环形缓冲 + 滑动窗口，
     为快路径（ms 级特征/触发）提供使能接口（v1 中为 TODO）。
  4. 默认采样率对齐数据契约 16000 SPS（v1 为 50000）。

参考: 西门子 SM1281 AFE 信号调理方法论（迁移自振动到电流域）
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from .filters import (
    remove_dc_offset,
    bandpass_filter,
    butter_bandpass_coefficients,
)
from .normalization import normalize_signal, NormalizationMethod
from .alignment import align_to_zero_crossing


@dataclass
class PreprocessConfig:
    """预处理配置参数（对齐《采集侧确认清单》数据契约）"""

    # ── 数据契约 ──
    sample_rate: float = 16000.0     # 采样率 (SPS/通道)
    channels: int = 3                # 通道数（三相 = 3）

    # ── 流式窗口（快路径） ──
    window_size: int = 256           # 流式窗口大小（采样点数）
    stride: int = 128                # 窗口步长（50% 重叠）

    # ── 直流偏置去除 ──
    dc_removal_enabled: bool = True
    dc_window: int = 100             # 滑动均值窗口

    # ── 带通滤波（VFD 友好，默认不杀诊断频带） ──
    filter_enabled: bool = True
    filter_order: int = 4            # 巴特沃斯阶数
    filter_lowcut: float = 5.0       # 低频截止：低于 VFD 最低输出频率
    filter_highcut: float = 4000.0   # 高频截止：不低于诊断频带上限

    # ── 归一化 ──
    norm_enabled: bool = True
    norm_method: str = "zscore"      # "zscore" | "minmax" | "rms" | "peak" | "none"

    # ── 相位对齐（默认关，电弧/单相分析用） ──
    alignment_enabled: bool = False
    alignment_tolerance: float = 0.02


class CurrentPreprocessor:
    """电流信号预处理器（支持单相 (N,) 与三相 (N, C)）

    用法:
        config = PreprocessConfig(sample_rate=16000, channels=3)
        pp = CurrentPreprocessor(config)
        clean = pp.process(raw_abc)           # (N, 3) → (N, 3)
        frames = pp.process_streaming(chunk)  # 快路径流式输出
    """

    def __init__(self, config: PreprocessConfig, sample_rate: Optional[float] = None):
        self.config = config
        self.sample_rate = sample_rate if sample_rate is not None else config.sample_rate
        self.channels = config.channels
        self.window_size = config.window_size
        self.stride = config.stride

        # 预计算滤波器系数（避免每帧重复计算）
        self._b = self._a = None
        if config.filter_enabled:
            self._b, self._a = butter_bandpass_coefficients(
                lowcut=config.filter_lowcut,
                highcut=config.filter_highcut,
                fs=self.sample_rate,
                order=config.filter_order,
            )

        # 流式缓冲
        self.reset()

    # ────────────────────────────────
    # 批处理
    # ────────────────────────────────
    def process(self, raw_samples: np.ndarray) -> np.ndarray:
        """执行完整预处理管线。

        Args:
            raw_samples: 原始采样数组，shape (N,) 单相 或 (N, C) 三相

        Returns:
            预处理后的干净信号，shape 同输入
        """
        x = np.asarray(raw_samples, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"不支持的数据维度: {x.ndim}")

        # 步骤1: 去除直流偏置（逐通道）
        if self.config.dc_removal_enabled:
            x = remove_dc_offset(x, window=self.config.dc_window)

        # 步骤2: 带通滤波（逐通道，截止由 VFD/诊断频带决定）
        if self.config.filter_enabled and self._b is not None:
            x = bandpass_filter(x, self._b, self._a)

        # 步骤3: 幅值归一化（逐通道）
        if self.config.norm_enabled:
            method = NormalizationMethod(self.config.norm_method)
            x = normalize_signal(x, method=method)

        # 步骤4: 相位对齐（逐通道，可选）
        if self.config.alignment_enabled:
            x = np.column_stack([
                align_to_zero_crossing(
                    x[:, ch], tolerance=self.config.alignment_tolerance
                )
                for ch in range(x.shape[1])
            ])

        return x[:, 0] if single else x

    # ────────────────────────────────
    # 流式处理（快路径：ms 级）
    # ────────────────────────────────
    def process_streaming(self, new_samples: np.ndarray) -> List[np.ndarray]:
        """流式处理：送入新采样，满窗口时输出处理结果帧。

        内部维护累积缓冲（环形语义），按 (window_size, stride) 滑动出帧，
        每帧输出 shape 与输入通道一致。

        Args:
            new_samples: 新到达采样，shape (M,) 或 (M, C)

        Returns:
            满窗口的处理结果帧列表（未满窗口时返回 []）
        """
        new = np.asarray(new_samples, dtype=np.float64)
        if new.ndim == 1:
            new = new[:, None]
        if new.ndim != 2 or new.shape[1] != self.channels:
            got = new.shape[1] if new.ndim == 2 else 1
            raise ValueError(f"通道数不匹配: 期望 {self.channels}，得到 {got}")

        self._buf = np.vstack([self._buf, new])
        frames: List[np.ndarray] = []
        while len(self._buf) >= self.window_size:
            win = self._buf[: self.window_size]
            self._buf = self._buf[self.stride:]
            frames.append(self.process(win))
        return frames

    def reset(self) -> None:
        """清空流式缓冲"""
        self._buf = np.empty((0, self.channels))


def create_preprocessor_from_yaml(
    config_dict: dict,
    sample_rate: Optional[float] = None,
) -> CurrentPreprocessor:
    """从 YAML 配置字典创建预处理器实例。"""
    pp_cfg = config_dict.get("preprocessing", {})
    sampling_cfg = config_dict.get("sampling", {})
    window_cfg = config_dict.get("window", {})

    cfg = PreprocessConfig(
        sample_rate=float(sampling_cfg.get("rate", sample_rate or 16000.0)),
        channels=int(pp_cfg.get("channels", 3)),
        window_size=int(window_cfg.get("size", 256)),
        stride=int(window_cfg.get("stride", 128)),
        dc_removal_enabled=pp_cfg.get("dc_removal", {}).get("enabled", True),
        dc_window=pp_cfg.get("dc_removal", {}).get("window", 100),
        filter_enabled=pp_cfg.get("bandpass_filter", {}).get("enabled", True),
        filter_order=pp_cfg.get("bandpass_filter", {}).get("order", 4),
        filter_lowcut=pp_cfg.get("bandpass_filter", {}).get("lowcut", 5.0),
        filter_highcut=pp_cfg.get("bandpass_filter", {}).get("highcut", 4000.0),
        norm_enabled=pp_cfg.get("normalization", {}).get("enabled", True),
        norm_method=pp_cfg.get("normalization", {}).get("method", "zscore"),
        alignment_enabled=pp_cfg.get("alignment", {}).get("enabled", False),
        alignment_tolerance=pp_cfg.get("alignment", {}).get("tolerance", 0.02),
    )
    return CurrentPreprocessor(cfg, sample_rate=cfg.sample_rate)
