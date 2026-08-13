"""
输入质量自评（10 鲁棒性 — R3/R4/R5/R6）
=========================================
在采集/特征之前对每帧做质量评估，把"输入劣化"与"真实故障"区分开：
削波、直流偏置、断线、丢样、缺相、相位错序必须被**标记**而非误诊为
堵转/谐波等负载故障。

对齐《08_鲁棒性/鲁棒性清单.md》目标矩阵：
  R3 削波/饱和 → 削波率监测（契约 full_scale），超阈标记"失真帧"
  R4 断线/直流偏置 → 极低方差 / 固定值 / 均值偏移检测
  R5 丢样 → 时戳连续性 + 丢样计数
  R6 缺相/相位错序 → 三相幅值 / 相序守卫

用法:
    iq = InputQuality(full_scale=3.3, channels=3)
    for raw_frame in stream:
        flags = iq.evaluate(raw_frame, fs=16000.0, t=t)
        if not flags.valid:
            log_quality(flags)   # 降级：不参与诊断
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .guards import safe_divide


@dataclass
class QualityFlags:
    """单帧输入质量标志"""

    valid: bool = True
    invalid: bool = False          # R1 NaN/Inf
    clipped: bool = False          # R3 削波/饱和
    dc_bias: bool = False          # R4 直流偏置
    dropout: bool = False          # R4 断线/常值（极低方差）
    sample_drop: bool = False      # R5 丢样（时戳不连续）
    phase_lost: bool = False       # R6 缺相
    phase_reversed: bool = False   # R6 相位错序
    score: float = 1.0             # 0..1 综合质量（1=正常）
    details: Dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        tags = [k for k in (
            "invalid", "clipped", "dc_bias", "dropout",
            "sample_drop", "phase_lost", "phase_reversed",
        ) if getattr(self, k)]
        return ",".join(tags) if tags else "ok"


class InputQuality:
    """输入质量自评器（无状态，每帧独立评估）。"""

    def __init__(
        self,
        channels: int = 3,
        full_scale: float = 1.0,          # 数据契约满量程（削波阈值基准）
        clip_margin: float = 0.02,        # 满量程容差比例
        clip_ratio_thresh: float = 0.02,  # 削波点占比阈值（> 判定削波）
        var_min: float = 1e-8,            # 低于此方差 → 断线/常值
        dc_bias_thresh: float = 0.5,      # |均值|/满量程 超过 → 直流偏置
        imbalance_thresh: float = 0.3,    # 某相 RMS 相对中位数偏离比例 → 缺相
        nominal_freq: float = 50.0,       # 基频（相位检测用）
    ):
        if channels < 2:
            # 相位/缺相检测需要多相；单通道仅做数值/削波/偏置/断线
            self.phase_supported = False
        else:
            self.phase_supported = True
        self.channels = channels
        self.full_scale = float(full_scale)
        self.clip_margin = float(clip_margin)
        self.clip_ratio_thresh = float(clip_ratio_thresh)
        self.var_min = float(var_min)
        self.dc_bias_thresh = float(dc_bias_thresh)
        self.imbalance_thresh = float(imbalance_thresh)
        self.nominal_freq = float(nominal_freq)
        self._prev_ts: Optional[float] = None

    def reset(self) -> None:
        """重置时戳状态（设备重启）"""
        self._prev_ts = None

    # ────────────────────────────
    # 主入口
    # ────────────────────────────
    def evaluate(
        self,
        frame: np.ndarray,
        fs: float = 16000.0,
        t: Optional[float] = None,
    ) -> QualityFlags:
        """评估一帧输入质量。

        Args:
            frame: (N,) 或 (N, C) 原始或预处理帧
            fs: 采样率 (Hz)
            t: 帧时间（秒），用于丢样检测（R5）

        Returns:
            QualityFlags
        """
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape
        flags = QualityFlags()

        # R1: 数值有效性
        if not np.all(np.isfinite(x)):
            flags.valid = False
            flags.invalid = True
            flags.score = 0.0
            flags.details["reason"] = "non_finite"
            self._prev_ts = t
            return flags

        # R3: 削波检测（max|·| 接近满量程的比例）
        clip_level = self.full_scale * (1.0 - self.clip_margin)
        clip_ratio = float(np.mean(np.max(np.abs(x), axis=1) >= clip_level))
        if clip_ratio > self.clip_ratio_thresh:
            flags.clipped = True
            flags.details["clip_ratio"] = clip_ratio

        # R4: 断线/常值（极低方差）
        ch_var = np.var(x, axis=0)
        if float(np.mean(ch_var)) < self.var_min:
            flags.dropout = True
            flags.details["var"] = float(np.mean(ch_var))

        # R4: 直流偏置（均值相对满量程）
        if self.full_scale > 0:
            mean_abs = float(np.mean(np.abs(np.mean(x, axis=0))))
            if mean_abs > self.dc_bias_thresh * self.full_scale:
                flags.dc_bias = True
                flags.details["dc"] = mean_abs

        # R5: 丢样（时戳不连续）—— 需调用方传入 t
        if t is not None and self._prev_ts is not None:
            dt = t - self._prev_ts
            expected = n / fs
            if dt > 2.0 * expected + 1e-6:
                flags.sample_drop = True
                flags.details["dt"] = dt
                flags.details["expected_dt"] = expected
        if t is not None:
            self._prev_ts = t

        # R6: 缺相 / 相位错序（需多相 + 足够幅值）
        if self.phase_supported and not flags.dropout:
            rms = np.sqrt(np.mean(x ** 2, axis=0))
            med = float(np.median(rms))
            if med > 1e-9:
                dev = np.abs(rms - med) / med
                if float(np.max(dev)) > self.imbalance_thresh:
                    flags.phase_lost = True
                    flags.details["rms"] = rms.tolist()
            self._check_phase_order(x, fs, flags)

        # 综合得分（从 1 起扣，供调用方分级降级）
        score = 1.0
        for flag, penalty in (
            ("invalid", 1.0), ("dropout", 0.9), ("dc_bias", 0.4),
            ("phase_lost", 0.5), ("phase_reversed", 0.4),
            ("sample_drop", 0.3), ("clipped", 0.2),
        ):
            if getattr(flags, flag):
                score -= penalty
        flags.score = float(max(0.0, score))

        # 硬无效：NaN/Inf 与断线（常值）→ 输入不可用，跳过诊断；
        # 其余（削波/偏置/丢样/缺相/错序）为软劣化，标记但可继续
        flags.valid = not flags.invalid and not flags.dropout
        if not flags.valid:
            flags.details.setdefault("reason", "input_unavailable")
        return flags

    # ────────────────────────────
    # 相位错序（R6）
    # ────────────────────────────
    def _check_phase_order(self, x: np.ndarray, fs: float, flags: QualityFlags) -> None:
        """基于基波 FFT 相位检测 A-B-C 相序（需 ≥ 1 个整周期帧）。"""
        n = x.shape[0]
        f = self.nominal_freq
        k = round(f * n / fs)
        if k < 2 or k >= n // 2:
            # 帧太短，无法分辨基波相位 → 不判定
            return
        fft = np.fft.rfft(x, axis=0)
        ph = np.angle(fft[k])                 # 每相基波相位
        dBA = ((ph[1] - ph[0] + np.pi) % (2 * np.pi)) - np.pi   # 相位差 → (-π, π]
        # 正序（A-B-C）：B 滞后 A 约 120° → dBA ≈ -2π/3
        # 错序（A-C-B）：B 超前 A 约 120° → dBA ≈ +2π/3
        # 用符号判定（远离 0）
        if abs(dBA) > np.pi / 3:              # 相移显著（非同相/异常）
            flags.phase_reversed = (dBA > 0)
        flags.details["phase_dba_deg"] = float(np.degrees(dBA))
