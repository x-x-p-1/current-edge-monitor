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
    SlowBaselineRemover,
    StreamingBandpass,
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

    # ── 直流偏置去除（v2.1：默认慢基线，修复滑动均值窗砍基波 bug） ──
    dc_removal_enabled: bool = True
    dc_method: str = "highpass"      # "highpass"/"slow_baseline"（推荐）| "sliding_mean"
    dc_cutoff_hz: float = 0.5        # 慢基线高通截止频率 (Hz)，仅 highpass 用
    dc_window: int = 100             # 滑动均值窗口（仅 sliding_mean 用，需 ≥ 数工频周期）

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

        # 流式缓冲 + 流式慢基线去直流 / 流式带通（状态延续）
        self._dc_stream: Optional[SlowBaselineRemover] = None
        self._bp_stream: Optional[StreamingBandpass] = None
        self.reset()

    # ────────────────────────────────
    # 批处理
    # ────────────────────────────────
    def _as_matrix(self, raw_samples: np.ndarray):
        """统一输入为 (N, C) 矩阵，返回 (x, single) 标记。"""
        x = np.asarray(raw_samples, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError(f"不支持的数据维度: {x.ndim}")
        return x, single

    def _apply_dc(self, x: np.ndarray) -> np.ndarray:
        """去除直流偏置（批处理：零相位，慢路径/分析用）。"""
        if not self.config.dc_removal_enabled:
            return x
        if self.config.dc_method in ("highpass", "slow_baseline"):
            # v2.1：慢基线估计，基波保留 ~100%（修复滑动均值窗砍基波 bug）
            return remove_dc_offset(
                x, method="highpass",
                cutoff_hz=self.config.dc_cutoff_hz, fs=self.sample_rate,
            )
        # 向后兼容：滑动均值法（窗口需 ≥ 数工频周期，见 filters.remove_dc_offset 说明）
        return remove_dc_offset(x, window=self.config.dc_window)

    def _bandpass(self, x: np.ndarray, stream: bool = False) -> np.ndarray:
        """带通滤波（逐通道）。stream=True 用状态延续因果滤波（快路径）。"""
        if not (self.config.filter_enabled and self._b is not None):
            return x
        if stream:
            if self._bp_stream is None:
                self._bp_stream = StreamingBandpass(
                    self._b, self._a, channels=x.shape[1])
            return self._bp_stream(x)
        return bandpass_filter(x, self._b, self._a)

    def _finish(self, x: np.ndarray, single: bool) -> np.ndarray:
        """幅值归一化 → 相位对齐，并还原单通道形状。

        注意：去直流与带通已在上游完成（批处理在 process / 流式在 _stream_filter），
        本方法只做非线性后处理（归一化/对齐），不引入额外频域畸变。
        """
        # 幅值归一化（逐通道）
        if self.config.norm_enabled:
            method = NormalizationMethod(self.config.norm_method)
            x = normalize_signal(x, method=method)

        # 相位对齐（逐通道，可选）
        if self.config.alignment_enabled:
            x = np.column_stack([
                align_to_zero_crossing(
                    x[:, ch], tolerance=self.config.alignment_tolerance
                )
                for ch in range(x.shape[1])
            ])

        return x[:, 0] if single else x

    def process(self, raw_samples: np.ndarray) -> np.ndarray:
        """执行完整预处理管线（批处理：去DC → 零相位带通 → 归一化/对齐）。

        Args:
            raw_samples: 原始采样数组，shape (N,) 单相 或 (N, C) 三相

        Returns:
            预处理后的干净信号，shape 同输入
        """
        x, single = self._as_matrix(raw_samples)
        x = self._apply_dc(x)
        x = self._bandpass(x, stream=False)
        return self._finish(x, single)

    # ────────────────────────────────
    # 流式处理（快路径：ms 级）
    # ────────────────────────────────
    def _stream_filter(self, x: np.ndarray) -> np.ndarray:
        """对**新增采样**做状态延续滤波（去DC → 带通）。

        状态只随输入流逐点前进（因果 IIR + zi 延续），不因窗口重叠回退，
        因此重叠帧只是从连续滤波输出里"切帧"，不会产生帧边界瞬态。
        """
        if self.config.dc_removal_enabled:
            if self.config.dc_method in ("highpass", "slow_baseline"):
                if self._dc_stream is None:
                    self._dc_stream = SlowBaselineRemover(
                        cutoff_hz=self.config.dc_cutoff_hz,
                        fs=self.sample_rate,
                        channels=self.channels,
                    )
                x = self._dc_stream(x)
            else:
                # sliding_mean：无状态滑动均值，逐块独立（低频近似）
                x = remove_dc_offset(x, window=self.config.dc_window)
        return self._bandpass(x, stream=True)

    def process_streaming(self, new_samples: np.ndarray) -> List[np.ndarray]:
        """流式处理：送入新采样，满窗口时输出处理结果帧。

        架构（v2.1）:
          原始采样逐块进入 → 状态延续因果滤波（去DC→带通）→ 连续滤波输出流
          → 按 (window_size, stride) 从滤波输出流切帧 → 归一化/对齐 → 帧

        与旧版（每帧独立 filtfilt）不同，滤波状态只随输入推进，
        重叠窗口（stride < window）不会导致因果滤波状态错位。

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

        # 1) 状态延续滤波（仅对新增采样）
        clean_new = self._stream_filter(new)
        self._buf_clean = np.vstack([self._buf_clean, clean_new])

        # 2) 从连续滤波输出流切帧（只切帧，不重滤波）
        frames: List[np.ndarray] = []
        while len(self._buf_clean) >= self.window_size:
            win = self._buf_clean[: self.window_size]
            self._buf_clean = self._buf_clean[self.stride:]
            frames.append(self._finish(win, single=False))
        return frames

    def reset(self) -> None:
        """清空流式缓冲与流式滤波状态"""
        self._buf_clean = np.empty((0, self.channels))
        if self._dc_stream is not None:
            self._dc_stream.reset()
        if self._bp_stream is not None:
            self._bp_stream.reset()


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
        dc_method=pp_cfg.get("dc_removal", {}).get("method", "highpass"),
        dc_cutoff_hz=pp_cfg.get("dc_removal", {}).get("cutoff_hz", 0.5),
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
