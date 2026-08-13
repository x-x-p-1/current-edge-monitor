"""
测试与验证 — 鲁棒性（10_鲁棒性 + 09 采集层集成）
==================================================
对齐《08_鲁棒性/鲁棒性清单.md》：
  R1 NaN/Inf 防护 · R2 除零守卫 · R3 削波 · R4 偏置/断线 · R5 丢样 ·
  R6 缺相/错序 · R10 单帧隔离 · §4.2 混沌注入。
"""
import importlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_rob = importlib.import_module("10_鲁棒性")
frame_is_valid = _rob.frame_is_valid
sanitize_nan_inf = _rob.sanitize_nan_inf
safe_divide = _rob.safe_divide
NumericGuard = _rob.NumericGuard
InputQuality = _rob.InputQuality
ChaosInjector = _rob.ChaosInjector

_pre = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _pre.CurrentPreprocessor
PreprocessConfig = _pre.PreprocessConfig

_acq = importlib.import_module("09_采集层")
WatchdogFeatures = _acq.WatchdogFeatures
TriggerEngine = _acq.TriggerEngine
SliceCapture = _acq.SliceCapture
CaptureContext = _acq.CaptureContext
AcquisitionEngine = _acq.AcquisitionEngine

FS = 16000.0


def _frame(rms_val: float = 1.0, n: int = 640, c: int = 3, seed: int = 0,
           noise: float = 0.01) -> np.ndarray:
    """整周期三相正弦帧（每通道目标 RMS）"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.column_stack([
        np.sin(2 * np.pi * 50 * t + ph)
        for ph in (0.0, -2 * np.pi / 3, 2 * np.pi / 3)
    ])
    x = x / np.sqrt(np.mean(x ** 2, axis=0)) * rms_val
    return x + noise * rng.standard_normal((n, c))


# ============================================================
# R1/R2 数值守卫
# ============================================================

class TestGuards(unittest.TestCase):
    """数值守卫：NaN/Inf 清洗、除零、对数"""

    def test_frame_is_valid(self):
        self.assertTrue(frame_is_valid(np.ones((8, 3))))
        self.assertFalse(frame_is_valid(np.array([[np.nan, 1.0, 2.0]])))
        self.assertFalse(frame_is_valid(np.array([[np.inf, 1.0, 2.0]])))

    def test_sanitize_nan_inf(self):
        x = np.array([[np.nan, 1.0, np.inf], [2.0, -np.inf, 3.0]])
        cleaned, has = sanitize_nan_inf(x, fill=0.0)
        self.assertTrue(has)
        self.assertTrue(np.all(np.isfinite(cleaned)))
        self.assertEqual(cleaned[0, 0], 0.0)
        # 无非有限 → has=False，原样返回
        y, h2 = sanitize_nan_inf(np.ones((4, 2)))
        self.assertFalse(h2)
        np.testing.assert_array_equal(y, np.ones((4, 2)))

    def test_safe_divide(self):
        num = np.array([1.0, 1.0, 1.0])
        den = np.array([2.0, 0.0, 1e-20])
        out = safe_divide(num, den)
        np.testing.assert_allclose(out, [0.5, 0.0, 0.0])  # 无 ±Inf/NaN
        self.assertTrue(np.all(np.isfinite(out)))

    def test_numeric_guard_degraded(self):
        guard = NumericGuard(max_invalid_streak=3)
        good = np.ones((8, 3))
        bad = np.full((8, 3), np.nan)
        self.assertTrue(guard.check(good)[1])
        self.assertFalse(guard.degraded)
        guard.check(bad)
        guard.check(bad)
        self.assertFalse(guard.degraded)
        guard.check(bad)
        self.assertTrue(guard.degraded)
        guard.check(good)
        self.assertFalse(guard.degraded)   # 恢复


# ============================================================
# R3/R4/R5/R6 输入质量自评
# ============================================================

class TestInputQuality(unittest.TestCase):
    """输入质量：削波 / 偏置 / 断线 / 丢样 / 缺相 / 错序"""

    def setUp(self):
        self.iq = InputQuality(channels=3, full_scale=1.5)

    def test_normal_frame_valid(self):
        flags = self.iq.evaluate(_frame(rms_val=1.0), fs=FS)
        self.assertTrue(flags.valid)
        self.assertGreater(flags.score, 0.9)

    def test_nan_invalid(self):
        x = _frame()
        x[3, 1] = np.nan
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.invalid)
        self.assertFalse(flags.valid)
        self.assertEqual(flags.score, 0.0)

    def test_clipped_detected(self):
        # 满量程 1.5：削波到 ±1.5，削波点占比超阈
        x = _frame(rms_val=2.0)          # 峰值 ~2.8 > 1.5
        x = np.clip(x, -1.5, 1.5)
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.clipped)

    def test_dropout_constant(self):
        x = np.full((320, 3), 2.5)       # 传感器断线：常值
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.dropout)
        self.assertFalse(flags.valid)

    def test_dc_bias_detected(self):
        x = _frame(rms_val=1.0) + 0.8    # 直流偏置 0.8 > 0.5*1.5
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.dc_bias)

    def test_sample_drop_detected(self):
        self.iq.evaluate(_frame(), fs=FS, t=0.0)
        flags = self.iq.evaluate(_frame(), fs=FS, t=0.5)   # 跳 0.5s
        self.assertTrue(flags.sample_drop)

    def test_phase_lost_detected(self):
        x = _frame()
        x[:, 1] *= 0.1                   # B 相几乎丢失
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.phase_lost)

    def test_phase_reversed_detected(self):
        x = _frame(n=640)                # 2 整周期
        x[:, [1, 2]] = x[:, [2, 1]]      # A-C-B 错序
        flags = self.iq.evaluate(x, fs=FS)
        self.assertTrue(flags.phase_reversed)


# ============================================================
# §4.2 混沌注入
# ============================================================

class TestChaosInjector(unittest.TestCase):
    """混沌注入器：各故障类型正确注入"""

    def setUp(self):
        self.chaos = ChaosInjector(seed=0, full_scale=1.5)
        self.base = _frame(rms_val=1.0, n=320)

    def test_all_data_kinds_return_array(self):
        for kind in ChaosInjector.DATA_KINDS:
            data, k = self.chaos.inject(self.base, kind=kind)
            self.assertEqual(k, kind)
            if kind != "sample_drop":
                self.assertIsInstance(data, np.ndarray)
            else:
                self.assertEqual(data.shape[0], 0)

    def test_nan_injection(self):
        data, k = self.chaos.inject(self.base, kind="nan")
        self.assertTrue(np.isnan(data).any())

    def test_inf_injection(self):
        data, k = self.chaos.inject(self.base, kind="inf")
        self.assertTrue(np.isinf(data).any())

    def test_dropout_injection(self):
        data, k = self.chaos.inject(self.base, kind="dropout")
        self.assertTrue(np.allclose(data, data[0:1]))   # 常值（每通道恒等）

    def test_clip_injection(self):
        x = _frame(rms_val=3.0)
        data, k = self.chaos.inject(x, kind="clip")
        self.assertLessEqual(np.max(np.abs(data)), 1.5)

    def test_phase_swap_injection(self):
        data, k = self.chaos.inject(self.base, kind="phase_swap")
        np.testing.assert_allclose(data[:, 1], self.base[:, 2])
        np.testing.assert_allclose(data[:, 2], self.base[:, 1])

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            self.chaos.inject(self.base, kind="bogus")


# ============================================================
# R10 单帧隔离 + 混沌端到端（engine 集成）
# ============================================================

class TestEngineRobustness(unittest.TestCase):
    """采集引擎鲁棒性：NaN 不崩溃、异常隔离、混沌全种类存活"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="rob_test_")
        self.pp = CurrentPreprocessor(
            PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
        self.iq = InputQuality(channels=3, full_scale=1.5)
        self.engine = AcquisitionEngine(
            self.pp,
            WatchdogFeatures(history=16),
            TriggerEngine(k_sigma=4.0, confirm_count=3, warmup_frames=30),
            SliceCapture(CaptureContext(
                pre_samples=1024, post_samples=1024, sample_rate=FS,
                channels=3, out_dir=self.tmpdir)),
            input_quality=self.iq,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nan_injection_does_not_crash(self):
        x = _frame(rms_val=1.0, n=320)
        x[::7, 1] = np.nan
        self.engine.feed(x)                       # 不应抛异常
        self.assertGreater(self.engine._nan_sanitized, 0)

    def test_non_array_input_isolated(self):
        self.engine.feed(None)                    # 突发异常
        self.assertEqual(self.engine._errors, 1)
        self.engine.feed(np.array([]))            # 空块
        self.assertGreaterEqual(self.engine._errors, 1)

    def test_dropout_frame_degraded_and_skipped(self):
        # 先喂正常帧（看门狗/触发推进）
        for _ in range(5):
            self.engine.feed(_frame(rms_val=1.0, n=320))
        frames_before = self.engine.wd.frame_idx
        # 断线段（常值）→ 质量门控跳过，看门狗不推进
        self.engine.feed(np.full((320, 3), 2.5))
        self.assertGreater(self.engine._degraded_frames, 0)
        self.assertEqual(self.engine.wd.frame_idx, frames_before)

    def test_chaos_all_kinds_survive(self):
        chaos = ChaosInjector(seed=1, full_scale=1.5)
        base = _frame(rms_val=1.0, n=320)
        for kind in ChaosInjector.DATA_KINDS:
            data, _ = chaos.inject(base, kind=kind)
            # 不应抛异常；exception 类型单独测
            self.engine.feed(data)
        self.engine.feed(None)                    # 突发异常
        # 进程存活 → 后续正常帧仍工作
        self.engine.feed(_frame(rms_val=1.0, n=320))
        self.assertGreaterEqual(self.engine.wd.frame_idx, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
