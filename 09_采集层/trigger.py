"""
触发引擎（M1 采集层 — stopping time）
======================================
实现"统计触发 + 去抖确认"的异常捕获触发（stopping time）：
设备大部分时间正常，异常是稀有事件；触发引擎在毫秒级看门狗特征流上
实时判定"此刻值得冻结切片"，驱动 slice_capture 落盘。

机制（复刻打底，P1 触发统计的工程近似）:
  1. 自适应基线：EMA 估计正常水平的均值 μ 与波动 σ（仅稳态帧更新基线，
     避免异常帧污染基线 —— 对齐 08 鲁棒性"状态机防漂移"）
  2. 统计判据：对平滑后的判据量 z = (x_ewma - μ) / σ，
     超过 ±K 触发候选（K 默认 4.0 → 正常误触率 ≈ 6e-5/帧，近似 ARL）
  3. EWMA 平滑判据量：抑制单帧毛刺（白噪声下误报率进一步下降）
  4. 去抖确认：连续 confirm_count 帧满足才真正触发；
     连续 release_count 帧不满足才复位（迟滞，防抖动）

数学保证（ARL）的严格推导属于 TODO E · P1 触发统计，本实现为工程基线。

用法:
    trig = TriggerEngine(k_sigma=4.0, confirm_count=3, release_count=5)
    for snap in watchdog_snapshots:
        ev = trig.update(snap, t_s=t)
        if ev.triggered:
            capture.trigger(...)
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .watchdog import WatchdogSnapshot


@dataclass
class TriggerEvent:
    """触发事件（单帧判定结果）"""

    triggered: bool                # 本帧是否进入触发状态（上升沿）
    active: bool                   # 当前是否处于触发激活态
    reason: str                    # 触发/维持原因（'normal' | 'rms' | 'crest' | ...）
    score: float                   # 归一化偏离度 z（|z| 越大越异常）
    frame_idx: int
    timestamp_s: float


class TriggerEngine:
    """统计触发引擎（EWMA 基线 + K·σ 判据 + 去抖确认）。"""

    # 支持的判据量
    METRICS = ("rms", "crest")

    def __init__(
        self,
        metric: str = "rms",
        k_sigma: float = 4.0,
        confirm_count: int = 3,
        release_count: int = 5,
        mu_alpha: float = 0.05,
        sigma_alpha: float = 0.02,
        smooth_alpha: float = 0.3,
        min_sigma: float = 1e-3,
        warmup_frames: int = 50,
    ):
        """
        Args:
            metric: 判据量，'rms'（平均 RMS）或 'crest'（平均峰值因子）
            k_sigma: 触发阈值（μ 的倍数 σ）
            confirm_count: 连续触发候选帧数达到才触发（去抖）
            release_count: 连续正常帧数达到才复位（迟滞）
            mu_alpha: 基线 μ 的 EMA 平滑系数（越小基线越稳）
            sigma_alpha: 波动 σ 的 EMA 平滑系数
            smooth_alpha: 判据量 EWMA 平滑系数（控单帧毛刺误报）
            min_sigma: σ 下限（防止除零 / 极平稳信号误触发）
            warmup_frames: 基线预热帧数（此前不触发）
        """
        if metric not in self.METRICS:
            raise ValueError(f"metric 必须 ∈ {self.METRICS}，得到 {metric}")
        if not (confirm_count >= 1 and release_count >= 1):
            raise ValueError("confirm_count / release_count 必须 ≥ 1")
        self.metric = metric
        self.k_sigma = k_sigma
        self.confirm_count = int(confirm_count)
        self.release_count = int(release_count)
        self.mu_alpha = mu_alpha
        self.sigma_alpha = sigma_alpha
        self.smooth_alpha = smooth_alpha
        self.min_sigma = min_sigma
        self.warmup_frames = int(warmup_frames)

        self.reset()

    def reset(self) -> None:
        """重置引擎状态（设备重启 / 长停后重新锚定）"""
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._x_smooth: Optional[float] = None
        self._cand_streak = 0      # 连续候选帧数
        self._normal_streak = 0    # 连续正常帧数
        self._active = False       # 当前是否触发激活
        self._last_active = False  # 上一帧激活态（用于上升沿判定）
        self.frame_idx = 0
        self._last_event: Optional[TriggerEvent] = None

    @property
    def baseline(self) -> tuple:
        """当前基线 (μ, σ)（诊断用）"""
        return (self._mu if self._mu is not None else 0.0,
                self._sigma if self._sigma is not None else 0.0)

    @property
    def last_event(self) -> Optional[TriggerEvent]:
        return self._last_event

    # ────────────────────────────
    # 判据量提取
    # ────────────────────────────
    def _metric_value(self, snap: WatchdogSnapshot) -> float:
        if self.metric == "rms":
            return float(np.mean(snap.rms))
        return float(np.mean(snap.crest_factor))

    # ────────────────────────────
    # 主入口
    # ────────────────────────────
    def update(
        self,
        snap: WatchdogSnapshot,
        t_s: Optional[float] = None,
    ) -> TriggerEvent:
        """输入一帧看门狗快照，输出触发判定。

        Args:
            snap: WatchdogSnapshot（来自 watchdog.update）
            t_s: 帧时间（秒），缺省用 snap.timestamp_s

        Returns:
            TriggerEvent
        """
        ts = float(t_s) if t_s is not None else snap.timestamp_s
        x = self._metric_value(snap)

        # EWMA 平滑判据量（控毛刺）
        if self._x_smooth is None:
            self._x_smooth = x
        else:
            self._x_smooth = (self.smooth_alpha * x
                              + (1.0 - self.smooth_alpha) * self._x_smooth)
        xs = self._x_smooth

        # ── 基线预热 ──
        if self._mu is None:
            self._mu = xs
            self._sigma = max(self.min_sigma, 1e-6)
        elif self.frame_idx < self.warmup_frames:
            # 预热期：仅更新基线，不触发
            self._mu = (1 - self.mu_alpha) * self._mu + self.mu_alpha * xs
            self._sigma = max(self.min_sigma,
                              (1 - self.sigma_alpha) * self._sigma
                              + self.sigma_alpha * abs(xs - self._mu))
        else:
            # 触发候选判据：平滑值（控单帧毛刺误报）
            z_smooth = (xs - self._mu) / max(self._sigma, self.min_sigma)
            candidate = abs(z_smooth) > self.k_sigma
            # 恢复判据：原始值（异常结束后立即回落 → 快速复位）
            z_raw = (x - self._mu) / max(self._sigma, self.min_sigma)
            recovered = abs(z_raw) < self.k_sigma

            if candidate:
                self._cand_streak += 1
            else:
                self._cand_streak = 0

            if not self._active:
                # 触发：连续候选达到 confirm_count（去抖）
                if self._cand_streak >= self.confirm_count:
                    self._active = True
                    self._normal_streak = 0
            else:
                # 复位：原始值连续恢复 release_count 帧（迟滞，快速复位）
                if recovered:
                    self._normal_streak += 1
                else:
                    self._normal_streak = 0
                if self._normal_streak >= self.release_count:
                    self._active = False
                    self._cand_streak = 0

            # 更新基线（仅稳态帧：非候选且非激活，避免异常帧污染基线）
            if not candidate and not self._active:
                self._mu = (1 - self.mu_alpha) * self._mu + self.mu_alpha * xs
                self._sigma = max(
                    self.min_sigma,
                    (1 - self.sigma_alpha) * self._sigma
                    + self.sigma_alpha * abs(xs - self._mu),
                )

        event = TriggerEvent(
            triggered=bool(self._active and self._last_active is False),
            active=self._active,
            reason=self._reason(),
            score=float((xs - self._mu) / max(self._sigma, self.min_sigma))
            if self._mu is not None else 0.0,
            frame_idx=self.frame_idx,
            timestamp_s=ts,
        )
        self._last_active = self._active
        self.frame_idx += 1
        self._last_event = event
        return event

    def _reason(self) -> str:
        if not self._active:
            return "normal"
        return f"{self.metric}_deviation"
