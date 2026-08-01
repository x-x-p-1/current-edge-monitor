"""
测试与验证 — 后处理与决策融合模块
"""

import sys
import os
import importlib
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_hyst = importlib.import_module("04_后处理与决策.hysteresis")
HysteresisAlarm = _hyst.HysteresisAlarm
MultiLevelHysteresisAlarm = _hyst.MultiLevelHysteresisAlarm
AlarmState = _hyst.AlarmState
EventAggregator = _hyst.EventAggregator

_df = importlib.import_module("04_后处理与决策.decision_fusion")
DecisionFusionEngine = _df.DecisionFusionEngine
FusionMethod = _df.FusionMethod
ModelPrediction = _df.ModelPrediction
ScoreSmoother = _df.ScoreSmoother


class TestHysteresisAlarm(unittest.TestCase):
    """迟滞报警测试"""

    def setUp(self):
        self.alarm = HysteresisAlarm(
            threshold_upper=0.8,
            threshold_lower=0.6,
            confirm_count=3,
            release_count=3,
        )

    def test_normal_no_trigger(self):
        """正常值不应触发报警"""
        for _ in range(10):
            state = self.alarm.update(0.2)
        self.assertEqual(state, AlarmState.NORMAL)

    def test_single_spike_no_alarm(self):
        """单次尖峰不应触发报警（防抖）"""
        self.alarm.update(0.2)
        self.alarm.update(0.9)  # 单次超阈值
        self.alarm.update(0.2)
        self.alarm.update(0.2)
        self.assertEqual(self.alarm.state, AlarmState.NORMAL)

    def test_continuous_trigger(self):
        """连续超阈值应触发报警"""
        for _ in range(3):
            self.alarm.update(0.85)
        self.assertEqual(self.alarm.state, AlarmState.ALARM)

    def test_release_after_alarm(self):
        """报警后连续低于释放阈值应解除"""
        # 触发报警
        for _ in range(3):
            self.alarm.update(0.85)
        self.assertEqual(self.alarm.state, AlarmState.ALARM)

        # 解除报警
        for _ in range(3):
            self.alarm.update(0.3)
        self.assertEqual(self.alarm.state, AlarmState.NORMAL)

    def test_no_release_during_alarm(self):
        """报警期间偶发低值不应解除"""
        # 触发报警
        for _ in range(3):
            self.alarm.update(0.85)
        self.assertEqual(self.alarm.state, AlarmState.ALARM)

        # 偶尔 1 帧低值
        self.alarm.update(0.3)
        self.assertEqual(self.alarm.state, AlarmState.ALARM)


class TestMultiLevelHysteresis(unittest.TestCase):
    """多级迟滞报警测试"""

    def setUp(self):
        self.alarm = MultiLevelHysteresisAlarm(
            warning_threshold=2.0,
            alarm_threshold=4.0,
            confirm_count=3,
            release_count=3,
            hysteresis_ratio=0.1,
        )

    def test_normal(self):
        """正常值 → NORMAL"""
        for _ in range(5):
            state = self.alarm.update(1.5)
        # 需要获取综合状态 - 这里用 update 返回值
        self.assertEqual(state, AlarmState.NORMAL)

    def test_warning(self):
        """预警值 → WARNING"""
        state = AlarmState.NORMAL
        for _ in range(5):
            state = self.alarm.update(2.5)
        self.assertEqual(state, AlarmState.WARNING)

    def test_alarm(self):
        """报警值 → ALARM"""
        state = AlarmState.NORMAL
        for _ in range(5):
            state = self.alarm.update(5.0)
        self.assertEqual(state, AlarmState.ALARM)


class TestEventAggregator(unittest.TestCase):
    """事件聚合器测试"""

    def setUp(self):
        self.aggregator = EventAggregator(merge_window_ms=500.0, frame_interval_ms=2.56)

    def test_single_event(self):
        """単一持续事件应合并为一个"""
        # 连续报警
        for _ in range(50):
            self.aggregator.update(is_alarm=True)

        events = self.aggregator.finalize()
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]["duration_frames"], 40)

    def test_separate_events(self):
        """间隔超过合并窗口的事件应分开"""
        # 事件1
        for _ in range(10):
            self.aggregator.update(is_alarm=True)

        # 长时间正常
        for _ in range(500):
            self.aggregator.update(is_alarm=False)

        # 事件2
        for _ in range(10):
            self.aggregator.update(is_alarm=True)

        events = self.aggregator.finalize()
        self.assertEqual(len(events), 2)


class TestDecisionFusion(unittest.TestCase):
    """决策融合测试"""

    def setUp(self):
        self.engine = DecisionFusionEngine(
            method=FusionMethod.WEIGHTED,
            weights={"arc_cnn": 0.5, "anomaly_ae": 0.3, "rule_based": 0.2},
        )

    def test_all_normal(self):
        """所有模型判断正常"""
        preds = [
            ModelPrediction("arc_cnn", False, 0.1, 0.1),
            ModelPrediction("anomaly_ae", False, 0.2, 1.5),
            ModelPrediction("rule_based", False, 0.3, 0.3),
        ]
        result = self.engine.fuse(preds)
        self.assertFalse(result.is_anomaly)

    def test_all_anomaly(self):
        """所有模型判断异常"""
        preds = [
            ModelPrediction("arc_cnn", True, 0.95, 0.95),
            ModelPrediction("anomaly_ae", True, 0.85, 4.5),
            ModelPrediction("rule_based", True, 0.80, 0.80),
        ]
        result = self.engine.fuse(preds)
        self.assertTrue(result.is_anomaly)
        self.assertGreater(result.overall_confidence, 0.8)


class TestScoreSmoother(unittest.TestCase):
    """分数平滑器测试"""

    def test_moving_average(self):
        smoother = ScoreSmoother(window_size=5, method="moving_average")
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        smoothed = [smoother.update(s) for s in scores]
        self.assertAlmostEqual(smoothed[-1], 0.7, places=5)

    def test_exponential(self):
        smoother = ScoreSmoother(window_size=10, method="exponential")
        for _ in range(5):
            smoother.update(0.5)
        result = smoother.update(0.5)
        self.assertAlmostEqual(result, 0.5, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
