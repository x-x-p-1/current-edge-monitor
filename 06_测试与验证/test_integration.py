"""
测试与验证 — 集成测试 (v2)

端到端的完整 v2 推理管线测试：
  模拟数据(00) → 预处理(01) → 快/慢特征(02) → 过程状态(03) → 迟滞/事件(04)

v1 版本用 256 点 @50k 单窗跑电弧管线（子周期窗口频域/峰值因子物理不成立），
v2 改为三相快/慢路径 + 过程状态 + 触发确认的端到端自检。
"""

import sys
import os
import importlib
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
Fault = _sim.Fault

_pp = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _pp.CurrentPreprocessor
PreprocessConfig = _pp.PreprocessConfig

_feat = importlib.import_module("02_特征提取")

_ps = importlib.import_module("03_检测模型.process_state")
ProcessStateClassifier = _ps.ProcessStateClassifier
StateRuleConfig = _ps.StateRuleConfig

_hyst = importlib.import_module("04_后处理与决策.hysteresis")
HysteresisAlarm = _hyst.HysteresisAlarm
EventAggregator = _hyst.EventAggregator
AlarmState = _hyst.AlarmState


FS = 16000.0
WIN, STRIDE = 256, 128
# 正常负载 RMS 标幺值（模拟器 amp=1.0）——部署时由标定 / P3 提供
BASE_LOAD_RMS = 0.71


class TestIntegrationV2(unittest.TestCase):
    """v2 端到端集成测试"""

    def setUp(self):
        # 快路径需幅值保持（去DC+带通，勿做单位方差归一化，否则状态机基线失准）
        self.pp = CurrentPreprocessor(
            PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS
        )

    def _frame_fast(self, sig, i):
        win = self.pp.process(sig[i:i + WIN])   # (WIN, 3)，幅值保持
        return _feat.extract_fast_features(win, FS)

    def test_normal_cycle_no_false_stall(self):
        """正常负载周期（空载→负载→空载）不应出现 STALL / 报警事件"""
        sig, _, _ = generate_dataset(
            duration=10.0, f1=50.0,
            load_profile=[(0.0, 2.0, 0.1), (2.0, 8.0, 1.0), (8.0, 10.0, 0.1)],
        )
        clf = ProcessStateClassifier(StateRuleConfig(initial_baseline=BASE_LOAD_RMS))
        stalls = 0
        for i in range(0, len(sig) - WIN, STRIDE):
            if clf.update(self._frame_fast(sig, i))["state"] == "STALL":
                stalls += 1
        self.assertEqual(stalls, 0, f"正常负载周期不应出现 STALL（出现 {stalls} 帧）")

    def test_stall_triggers_event(self):
        """堵转注入：状态机检出 STALL，迟滞 + 事件聚合产出报警事件"""
        sig, _, _ = generate_dataset(
            duration=8.0, f1=50.0,
            load_profile=[(0.0, 8.0, 1.0)],
            faults=[Fault(kind="stall", start=5.0, dur=1.0, depth=1.8)],
        )
        clf = ProcessStateClassifier(
            StateRuleConfig(initial_baseline=BASE_LOAD_RMS, stall_confirm=3)
        )
        alarm = HysteresisAlarm(
            threshold_upper=0.6, threshold_lower=0.4,
            confirm_count=3, release_count=3,
        )
        agg = EventAggregator(merge_window_ms=500.0, frame_interval_ms=8.0)

        stall_seen = False
        for i in range(0, len(sig) - WIN, STRIDE):
            res = clf.update(self._frame_fast(sig, i))
            if res["state"] == "STALL":
                stall_seen = True
            # 疑似异常（STALL/TRANSIENT）→ 高分喂迟滞
            score = 0.9 if res["state"] in ("STALL", "TRANSIENT") else 0.1
            state = alarm.update(score)
            agg.update(state == AlarmState.ALARM)

        self.assertTrue(stall_seen, "堵转应被状态机检出")
        events = agg.finalize()
        self.assertGreater(len(events), 0, "堵转应触发至少 1 个报警事件")

    def test_slow_features_run(self):
        """慢路径特征在长窗上可运行（三相 + MCSA 边带）"""
        sig, _, _ = generate_dataset(
            duration=4.0, f1=50.0,
            faults=[Fault(kind="rotor_sideband", start=1.0, dur=2.0,
                          slip=0.03, depth=0.05)],
        )
        slow = _feat.extract_slow_features(sig, FS, f1=50.0, slip=0.03)
        self.assertIn("ch0_sideband_ratio", slow)
        self.assertTrue(any(k.startswith("3p_") for k in slow), "应含三相跨相特征")

    def test_timing_constraint(self):
        """快路径时序约束：预处理 + 快特征 + 状态更新应 < 5ms/帧（ms 级）"""
        import time

        sig, _, _ = generate_dataset(duration=2.0, f1=50.0,
                                     load_profile=[(0.0, 2.0, 1.0)])
        clf = ProcessStateClassifier(StateRuleConfig(initial_baseline=BASE_LOAD_RMS))

        total_times = []
        for i in range(0, len(sig) - WIN, STRIDE):
            start = time.perf_counter()
            clf.update(self._frame_fast(sig, i))
            total_times.append((time.perf_counter() - start) * 1_000_000)

        total_times = np.array(total_times)
        mean_us = np.mean(total_times)
        p99_us = np.percentile(total_times, 99)

        print(f"[TIMING] 快路径 Mean: {mean_us:.1f} µs, P99: {p99_us:.1f} µs")
        print(f"         目标: < 5000 µs (5ms 预处理+快特征+状态更新)")

        self.assertLess(mean_us, 5000,
                       f"快路径应 < 5ms/帧，实际 {mean_us:.0f}µs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
