"""
过程状态识别 (v2 新增)
=====================
基于规则基线的"过程状态"分类器（快速路径，无需训练）——作为"复刻打底"的
过程管理状态层，输出给触发引擎（04 后处理）与状态上下文。

状态集合（三相变频电机）：
  STOP       停机 / 无电流
  TRANSIENT  启动 / 负载突变瞬态
  IDLE       空载
  LOAD       负载运行
  STALL      堵转 / 过流
  UNKNOWN    未判定

判定依据（快路径特征，见 02 `cadence.extract_fast_features`）：
  - 相 RMS 均值 vs 自适应基线（DKW 思路）：空载 / 负载 / 堵转的相对比值
  - 帧间 RMS 突变率：瞬态
  - 堵转需连续确认 N 帧（对齐 04 的迟滞/确认机制）

⚠️ 说明：自适应基线的"正常建模"终极形态由数学家 P3（无监督正常建模）接替；
  本模块是规则版基线（复刻打底），保证无标签阶段即可运行。基线初始化依赖
  首帧，更稳健的基线应来自额定电流标定或 P3 模型。
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ProcessState(str, Enum):
    STOP = "STOP"
    TRANSIENT = "TRANSIENT"
    IDLE = "IDLE"
    LOAD = "LOAD"
    STALL = "STALL"
    UNKNOWN = "UNKNOWN"


@dataclass
class StateRuleConfig:
    # 停机：相 RMS 均值低于此值（标幺，需在归一化/标定后信号上运行）
    stop_rms: float = 0.05
    # 空载：RMS/基线 ≤ 此比例
    idle_ratio: float = 0.35
    # 负载：RMS/基线 ≥ 此比例
    load_ratio_min: float = 0.50
    # 堵转：RMS/基线 ≥ 此比例
    stall_ratio: float = 1.60
    # 瞬态：帧间 RMS 相对突变率 ≥ 此值
    transient_dratio: float = 0.40
    # 基线 EWMA 更新系数（小 → 慢，避免被异常污染）
    baseline_alpha: float = 0.02
    # 堵转连续确认帧数
    stall_confirm: int = 3
    # 通道数（三相 = 3）
    channels: int = 3
    # 初始基线（正常负载 RMS，来自额定电流标定或 P3 模型）；
    # 为 None 时取首帧（首帧可能为空载，导致负载被误判为堵转——部署时应给定）
    initial_baseline: Optional[float] = None


class ProcessStateClassifier:
    """规则基线过程状态分类器（快路径，状态机）"""

    def __init__(self, config: Optional[StateRuleConfig] = None):
        self.cfg = config or StateRuleConfig()
        self._baseline: Optional[float] = self.cfg.initial_baseline
        self._prev_rms: Optional[float] = None
        self._stall_cnt: int = 0
        self._state: ProcessState = ProcessState.UNKNOWN

    def set_baseline(self, value: float) -> None:
        """设定基线（正常负载 RMS）。部署时应在额定/标定工况下调用。"""
        self._baseline = float(value)

    def reset(self) -> None:
        self._baseline = None
        self._prev_rms = None
        self._stall_cnt = 0
        self._state = ProcessState.UNKNOWN

    # ────────────────────────────────
    def update(self, fast_features: Dict[str, float]) -> Dict:
        """输入一帧快路径特征，更新状态并返回结果。

        Args:
            fast_features: `02.cadence.extract_fast_features` 的输出
                （含 ch{c}_rms 等键）

        Returns:
            {state, rms_mean, baseline, ratio, stall_count, confidence}
        """
        rms = float(np.mean([
            fast_features.get(f"ch{c}_rms", 0.0)
            for c in range(self.cfg.channels)
        ]))
        eps = 1e-9

        # 1) 停机
        if rms < self.cfg.stop_rms:
            self._stall_cnt = 0
            self._state = ProcessState.STOP
            self._prev_rms = rms
            return self._result(rms, 0.0, 0.95)

        # 2) 瞬态（帧间突变）
        if self._prev_rms is not None:
            d = abs(rms - self._prev_rms) / max(self._prev_rms, eps)
            if d >= self.cfg.transient_dratio:
                self._stall_cnt = 0
                self._state = ProcessState.TRANSIENT
                self._prev_rms = rms
                return self._result(rms, 0.0, min(1.0, d / self.cfg.transient_dratio))

        # 3) 基线初始化 / 更新
        if self._baseline is None:
            self._baseline = rms
        ratio = rms / max(self._baseline, eps)

        # 4) 状态判定
        if ratio >= self.cfg.stall_ratio:
            self._stall_cnt += 1
            if self._stall_cnt >= self.cfg.stall_confirm:
                self._state = ProcessState.STALL
                conf = min(1.0, 0.7 + 0.1 * self._stall_cnt)
            else:
                self._state = ProcessState.TRANSIENT  # 疑似堵转，等确认
                conf = 0.4
        else:
            self._stall_cnt = 0
            if ratio <= self.cfg.idle_ratio:
                self._state = ProcessState.IDLE
                conf = min(1.0, (self.cfg.idle_ratio - ratio) / self.cfg.idle_ratio + 0.5)
            elif ratio >= self.cfg.load_ratio_min:
                self._state = ProcessState.LOAD
                conf = min(1.0, 0.6 + (ratio - self.cfg.load_ratio_min))
            else:
                self._state = ProcessState.UNKNOWN
                conf = 0.3

        # 5) 基线更新（仅 LOAD 态；IDLE 更新会把基线拉低，导致下次负载误判为堵转）
        if self._state is ProcessState.LOAD and self._baseline is not None:
            a = self.cfg.baseline_alpha
            self._baseline = (1.0 - a) * self._baseline + a * rms

        self._prev_rms = rms
        return self._result(rms, ratio, conf)

    # ────────────────────────────────
    def _result(self, rms: float, ratio: float, conf: float) -> Dict:
        return {
            "state": self._state.value,
            "rms_mean": round(float(rms), 6),
            "baseline": round(float(self._baseline if self._baseline is not None else 0.0), 6),
            "ratio": round(float(ratio), 6),
            "stall_count": self._stall_cnt,
            "confidence": round(float(min(max(conf, 0.0), 1.0)), 4),
        }


def classify_state(
    fast_features: Dict[str, float],
    baseline: float,
    config: Optional[StateRuleConfig] = None,
) -> Dict:
    """纯函数版单帧判定（不维护状态机，基线由调用方给定/维护）。"""
    cfg = config or StateRuleConfig()
    rms = float(np.mean([
        fast_features.get(f"ch{c}_rms", 0.0)
        for c in range(cfg.channels)
    ]))
    eps = 1e-9
    if rms < cfg.stop_rms:
        return {"state": ProcessState.STOP.value, "rms_mean": rms, "ratio": 0.0}
    ratio = rms / max(baseline, eps)
    if ratio >= cfg.stall_ratio:
        state = ProcessState.STALL.value
    elif ratio <= cfg.idle_ratio:
        state = ProcessState.IDLE.value
    elif ratio >= cfg.load_ratio_min:
        state = ProcessState.LOAD.value
    else:
        state = ProcessState.UNKNOWN.value
    return {"state": state, "rms_mean": float(rms), "ratio": round(float(ratio), 6)}
