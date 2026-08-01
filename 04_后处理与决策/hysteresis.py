"""
迟滞报警管理模块

借鉴西门子 APL MonAnL 功能块的迟滞与连续确认报警机制。

核心思想（来自西门子 Condition Monitoring 工程体系）:
  1. 连续 N 次确认才触发报警（防止单帧噪声误报）
  2. 迟滞量 H 确保报警状态稳定（防止阈值边界抖动）
  3. 确认释放也需连续 M 次（防止报警状态振荡）

数学表达:
    State_k = {
        ALARM,   if Σ 连续确认帧 >= N_confirm
        NORMAL,  if Σ 连续释放帧 >= N_release
        State_k-1, otherwise  (保持上一帧状态)
    }

应用场景:
  - 电弧检测: 需要连续 3 帧确认（约 7.68ms）才触发断路
  - 电能质量: THD 超标连续 5 帧才报警
  - 过流保护: 即时响应（N_confirm=1）
"""

from enum import Enum
from typing import Dict, Optional, Tuple


class AlarmState(str, Enum):
    """报警状态"""
    NORMAL = "normal"       # 正常
    WARNING = "warning"     # 预警
    ALARM = "alarm"         # 报警


class HysteresisAlarm:
    """
    迟滞报警管理器

    每个检测指标一个实例，维护独立的计数器和状态机。

    用法:
        alarm = HysteresisAlarm(
            threshold_upper=0.85,
            threshold_lower=0.70,
            confirm_count=3,
            release_count=5,
        )

        for score in detection_scores:
            state = alarm.update(score)
            if state == AlarmState.ALARM:
                trigger_protection()
    """

    def __init__(
        self,
        threshold_upper: float,
        threshold_lower: Optional[float] = None,
        confirm_count: int = 3,
        release_count: int = 5,
    ):
        """
        Args:
            threshold_upper: 报警触发阈值 (高于此值开始计数)
            threshold_lower: 报警解除阈值 (低于此值开始计数)
                             如果为 None，则使用 threshold_upper * 0.85
            confirm_count: 连续确认帧数（触发报警所需）
            release_count: 连续确认释放帧数（解除报警所需）
        """
        self.threshold_upper = threshold_upper
        self.threshold_lower = (
            threshold_lower if threshold_lower is not None
            else threshold_upper * 0.85
        )
        self.confirm_count = confirm_count
        self.release_count = release_count

        # 状态机
        self._state = AlarmState.NORMAL
        self._confirm_counter = 0   # 连续超阈值计数
        self._release_counter = 0   # 连续低于释放阈值计数

        # 历史
        self._score_history = []
        self._state_history = []

    @property
    def state(self) -> AlarmState:
        return self._state

    @property
    def is_alarm(self) -> bool:
        return self._state == AlarmState.ALARM

    @property
    def is_warning(self) -> bool:
        return self._state == AlarmState.WARNING

    def update(self, score: float) -> AlarmState:
        """
        输入当前帧的检测分数，更新状态机并返回当前状态。

        Args:
            score: 当前帧的检测分数（0~1 或任意阈值比较值）

        Returns:
            当前的 AlarmState
        """
        self._score_history.append(score)

        # ── 状态转移逻辑 ──

        if self._state == AlarmState.NORMAL:
            # 检查是否应该进入报警状态
            if score > self.threshold_upper:
                self._confirm_counter += 1
                self._release_counter = 0
            else:
                self._confirm_counter = max(0, self._confirm_counter - 1)

            # 达到确认帧数 → 直接报警
            if self._confirm_counter >= self.confirm_count:
                self._state = AlarmState.ALARM

        elif self._state == AlarmState.ALARM:
            # 检查是否应该解除报警
            if score < self.threshold_lower:
                self._release_counter += 1
            else:
                self._release_counter = 0

            # 达到释放帧数 → 解除
            if self._release_counter >= self.release_count:
                self._state = AlarmState.NORMAL
                self._confirm_counter = 0

        self._state_history.append(self._state)
        return self._state

    def reset(self):
        """重置状态机"""
        self._state = AlarmState.NORMAL
        self._confirm_counter = 0
        self._release_counter = 0

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "state": self._state.value,
            "confirm_counter": self._confirm_counter,
            "release_counter": self._release_counter,
            "total_alarms": sum(
                1 for s in self._state_history if s == AlarmState.ALARM
            ),
        }


class MultiLevelHysteresisAlarm:
    """
    多级迟滞报警（正常 → 预警 → 报警）

    类似西门子 CMS2000 的三级状态:
      - Zone A (OK): DKW ≤ 2.0 → NORMAL
      - Zone B (Warning): 2.0 < DKW ≤ 4.0 → WARNING
      - Zone C (Alarm): DKW > 4.0 → ALARM
    """

    def __init__(
        self,
        warning_threshold: float,
        alarm_threshold: float,
        confirm_count: int = 3,
        release_count: int = 5,
        hysteresis_ratio: float = 0.10,
    ):
        """
        Args:
            warning_threshold: 预警阈值
            alarm_threshold: 报警阈值（应 > warning_threshold）
            confirm_count: 连续确认次数
            release_count: 连续释放次数
            hysteresis_ratio: 迟滞比例（0.05~0.10）
        """
        self.warning_threshold = warning_threshold
        self.alarm_threshold = alarm_threshold
        self.confirm_count = confirm_count
        self.release_count = release_count
        self.hysteresis_ratio = hysteresis_ratio

        # 为每级创建子报警器
        self._warning_alarm = HysteresisAlarm(
            threshold_upper=warning_threshold,
            threshold_lower=warning_threshold * (1 - hysteresis_ratio),
            confirm_count=confirm_count,
            release_count=release_count,
        )
        self._alarm_alarm = HysteresisAlarm(
            threshold_upper=alarm_threshold,
            threshold_lower=alarm_threshold * (1 - hysteresis_ratio),
            confirm_count=confirm_count,
            release_count=release_count,
        )

    def update(self, score: float) -> AlarmState:
        """
        更新多级状态

        Returns:
            综合状态
        """
        alarm_state = self._alarm_alarm.update(score)
        warning_state = self._warning_alarm.update(score)

        if alarm_state == AlarmState.ALARM:
            return AlarmState.ALARM
        elif warning_state == AlarmState.ALARM:
            return AlarmState.WARNING
        else:
            return AlarmState.NORMAL

    @property
    def state(self) -> AlarmState:
        return self._state if hasattr(self, '_state') else AlarmState.NORMAL

    def reset(self):
        self._warning_alarm.reset()
        self._alarm_alarm.reset()


# ============================================================
# 事件聚合器
# ============================================================

class EventAggregator:
    """
    事件聚合器 — 将连续的短时报警帧合并为单次事件

    工业现场中，一次物理故障可能触发持续数百毫秒的报警帧。
    如果不做聚合，会记录大量重复报警，产生"报警风暴"。

    聚合策略:
        如果两次报警之间的间隔 < merge_window_ms，
        则合并为同一事件。
    """

    def __init__(self, merge_window_ms: float = 500.0, frame_interval_ms: float = 2.56):
        """
        Args:
            merge_window_ms: 合并窗口 (毫秒)
            frame_interval_ms: 帧间隔 (毫秒)
        """
        self.merge_window_ms = merge_window_ms
        self.frame_interval_ms = frame_interval_ms
        self.merge_window_frames = int(merge_window_ms / frame_interval_ms)

        self._events = []
        self._current_event = None
        self._frame_idx = 0
        self._last_alarm_frame = -9999  # 上一次报警的帧号

    def update(self, is_alarm: bool, metadata: Optional[dict] = None):
        """
        输入每帧的报警状态

        Args:
            is_alarm: 当前帧是否报警
            metadata: 附加信息
        """
        self._frame_idx += 1

        if is_alarm:
            # 检查是否应该合并到当前事件
            if (
                self._current_event is not None
                and (self._frame_idx - self._last_alarm_frame) <= self.merge_window_frames
            ):
                # 合并: 更新结束帧
                self._current_event["end_frame"] = self._frame_idx
                self._current_event["duration_frames"] += 1
            else:
                # 新事件
                if self._current_event is not None:
                    self._events.append(self._current_event)

                self._current_event = {
                    "start_frame": self._frame_idx,
                    "end_frame": self._frame_idx,
                    "duration_frames": 1,
                    "start_time_ms": self._frame_idx * self.frame_interval_ms,
                    "metadata": metadata or {},
                }

            self._last_alarm_frame = self._frame_idx

    def finalize(self) -> list:
        """结束并返回所有事件"""
        if self._current_event is not None:
            self._events.append(self._current_event)
            self._current_event = None

        return self._events

    @property
    def event_count(self) -> int:
        return len(self._events)

    def get_latest_events(self, n: int = 10) -> list:
        """返回最近 N 个事件"""
        return self._events[-n:] if self._events else []

    def reset(self):
        self._events = []
        self._current_event = None
        self._frame_idx = 0
        self._last_alarm_frame = -9999
