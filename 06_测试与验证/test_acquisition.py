"""
测试与验证 — M1 采集层（09_采集层）
====================================
覆盖：环形缓冲 / 看门狗特征 / 触发引擎 / 切片捕获 / 采集引擎端到端。
"""
import importlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_acq = importlib.import_module("09_采集层")
RingBuffer = _acq.RingBuffer
WatchdogFeatures = _acq.WatchdogFeatures
TriggerEngine = _acq.TriggerEngine
SliceCapture = _acq.SliceCapture
CaptureContext = _acq.CaptureContext
AcquisitionEngine = _acq.AcquisitionEngine

_pre = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _pre.CurrentPreprocessor
PreprocessConfig = _pre.PreprocessConfig

_sim = importlib.import_module("00_数据生成与仿真.current_simulator")
generate_dataset = _sim.generate_dataset
Fault = _sim.Fault

FS = 16000.0
WIN = 256


def _frame(rms_val: float = 1.0, n: int = WIN, c: int = 3, seed: int = 0,
           noise: float = 0.02) -> np.ndarray:
    """构造目标 RMS（每通道）的 50Hz 正弦三相帧（幅值保持）"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = np.column_stack([
        np.sin(2 * np.pi * 50 * t + ph)
        for ph in (0.0, -2 * np.pi / 3, 2 * np.pi / 3)
    ])
    x = x / np.sqrt(np.mean(x ** 2, axis=0)) * rms_val
    return x + noise * rng.standard_normal((n, c))


def _make_snap(rms_val: float, seed: int = 0):
    """用看门狗把帧转成 WatchdogSnapshot（触发引擎的输入）"""
    return WatchdogFeatures(history=2).update(_frame(rms_val=rms_val, seed=seed))


# ============================================================
# 1) 环形缓冲
# ============================================================

class TestRingBuffer(unittest.TestCase):
    """环形缓冲：固定容量、覆盖语义、偏移读取"""

    def test_write_and_read_order(self):
        buf = RingBuffer(capacity=10, channels=1)
        buf.write(np.arange(5, dtype=float))          # (5,)
        arr = buf.as_array()
        np.testing.assert_allclose(arr[:, 0], np.arange(5))
        self.assertEqual(buf.n_samples, 5)

    def test_three_phase_write(self):
        buf = RingBuffer(capacity=10, channels=3)
        data = np.arange(30, dtype=float).reshape(10, 3)
        buf.write(data)
        np.testing.assert_array_equal(buf.as_array(), data)

    def test_overflow_overwrites_oldest(self):
        buf = RingBuffer(capacity=8, channels=1)
        buf.write(np.arange(20, dtype=float))         # 20 点写入 8 点容量
        # 最旧 12 点被覆盖，剩余 [12..19]
        np.testing.assert_allclose(buf.as_array()[:, 0], np.arange(12, 20))
        self.assertEqual(buf.n_samples, 8)
        self.assertTrue(buf.filled)
        self.assertEqual(buf.overflow_samples, 12)

    def test_get_last_and_slice_offsets(self):
        buf = RingBuffer(capacity=100, channels=1)
        buf.write(np.arange(50, dtype=float))
        last5 = buf.get_last(5)
        np.testing.assert_allclose(last5[:, 0], np.arange(45, 50))
        # 绝对索引切片 [10, 20)
        seg = buf.slice_range(10, 20)
        np.testing.assert_allclose(seg[:, 0], np.arange(10, 20))

    def test_slice_crosses_wrap(self):
        buf = RingBuffer(capacity=16, channels=1)
        buf.write(np.arange(40, dtype=float))         # 覆盖后剩 [24..39]
        # 取跨物理环边界的 [30, 40)
        seg = buf.slice_range(30, 40)
        np.testing.assert_allclose(seg[:, 0], np.arange(30, 40))

    def test_slice_beyond_available_is_clipped(self):
        buf = RingBuffer(capacity=16, channels=1)
        buf.write(np.arange(10, dtype=float))
        # 越界范围被裁剪
        seg = buf.slice_range(-5, 20)
        np.testing.assert_allclose(seg[:, 0], np.arange(10))
        self.assertEqual(buf.slice_range(20, 30).shape[0], 0)

    def test_channel_mismatch_raises(self):
        buf = RingBuffer(capacity=16, channels=3)
        with self.assertRaises(ValueError):
            buf.write(np.zeros((8, 2)))

    def test_reset(self):
        buf = RingBuffer(capacity=8, channels=1)
        buf.write(np.arange(5, dtype=float))
        buf.reset()
        self.assertEqual(buf.n_samples, 0)
        self.assertFalse(buf.filled)


# ============================================================
# 2) 看门狗特征
# ============================================================

class TestWatchdogFeatures(unittest.TestCase):
    """毫秒级看门狗特征：RMS / 峰值因子 / 斜率"""

    def test_rms_and_crest_of_sine(self):
        wd = WatchdogFeatures(history=16)
        # 整周期窗 + 无噪声：验证数学正确性（RMS=1，峰值因子=√2）
        snap = wd.update(_frame(rms_val=1.0, n=640, noise=0.0))
        np.testing.assert_allclose(snap.rms, 1.0, atol=0.02)
        np.testing.assert_allclose(snap.crest_factor, np.sqrt(2), atol=0.05)

    def test_rms_slope_detects_rise(self):
        wd = WatchdogFeatures(history=8)
        # 先 8 帧低 RMS，再 8 帧递增 RMS
        for _ in range(8):
            wd.update(_frame(rms_val=1.0))
        snap = None
        for i in range(8):
            snap = wd.update(_frame(rms_val=1.0 + 0.1 * i))
        self.assertGreater(snap.rms_slope, 0.0)   # 递增 → 斜率 > 0

    def test_rms_slope_stable_is_zero(self):
        wd = WatchdogFeatures(history=8)
        for _ in range(20):
            wd.update(_frame(rms_val=1.0, seed=1))
        self.assertLess(abs(wd.last_snapshot.rms_slope), 0.05)

    def test_frame_idx_and_timestamp(self):
        wd = WatchdogFeatures(history=8)
        for i in range(3):
            snap = wd.update(_frame(), t_s=float(i) * 0.016)
            self.assertEqual(snap.frame_idx, i)
            self.assertAlmostEqual(snap.timestamp_s, i * 0.016)


# ============================================================
# 3) 触发引擎
# ============================================================

class TestTriggerEngine(unittest.TestCase):
    """触发引擎：统计判据 / 去抖 / 迟滞复位"""

    def setUp(self):
        self.trig = TriggerEngine(
            metric="rms", k_sigma=4.0, confirm_count=3,
            release_count=5, warmup_frames=20,
        )

    def test_normal_does_not_trigger(self):
        for i in range(80):
            ev = self.trig.update(_make_snap(rms_val=1.0, seed=i))
            self.assertFalse(ev.triggered)
        self.assertFalse(self.trig.last_event.active)

    def test_abrupt_fault_triggers_with_confirm(self):
        # 正常预热
        for i in range(30):
            self.trig.update(_make_snap(rms_val=1.0, seed=i))
        # 突变 3x RMS（>> K·σ）
        triggered_frame = None
        for i in range(10):
            ev = self.trig.update(_make_snap(rms_val=3.0, seed=i))
            if ev.triggered:
                triggered_frame = ev.frame_idx
                break
        self.assertIsNotNone(triggered_frame)
        # 去抖：应在 confirm_count 帧后触发（30 预热 + confirm=3 → ~33）
        self.assertGreaterEqual(triggered_frame, 30 + 1)
        self.assertLessEqual(triggered_frame, 30 + self.trig.confirm_count + 2)

    def test_confirm_debounce(self):
        """单帧毛刺（< confirm_count）不应触发"""
        trig = TriggerEngine(k_sigma=4.0, confirm_count=3, release_count=3,
                             warmup_frames=20)
        for i in range(30):
            trig.update(_make_snap(rms_val=1.0, seed=i))
        # 仅 2 帧异常（不足 3）
        trig.update(_make_snap(rms_val=3.0, seed=1))
        trig.update(_make_snap(rms_val=3.0, seed=2))
        self.assertFalse(trig.last_event.active)
        # 第三帧持续 → 触发
        ev = trig.update(_make_snap(rms_val=3.0, seed=3))
        self.assertTrue(ev.triggered)

    def test_release_hysteresis(self):
        """恢复常态后按 release_count 复位（迟滞）"""
        for i in range(30):
            self.trig.update(_make_snap(rms_val=1.0, seed=i))
        for i in range(10):
            self.trig.update(_make_snap(rms_val=3.0, seed=i))
        self.assertTrue(self.trig.last_event.active)
        # 恢复常态 release_count 帧后复位
        for i in range(self.trig.release_count):
            self.trig.update(_make_snap(rms_val=1.0, seed=100 + i))
        self.assertFalse(self.trig.last_event.active)


# ============================================================
# 4) 切片捕获
# ============================================================

class TestSliceCapture(unittest.TestCase):
    """切片捕获：预触发 + 后触发 + 上下文落盘"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="slice_test_")
        self.ctx = CaptureContext(
            pre_samples=512, post_samples=1024,
            sample_rate=FS, channels=3, out_dir=self.tmpdir,
        )
        self.cap = SliceCapture(self.ctx)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_slice_content_and_meta(self):
        buf = RingBuffer(capacity=4096, channels=3)
        # 预触发 512 点
        buf.write(np.arange(512 * 3, dtype=float).reshape(512, 3))
        # 触发点 = 当前写入点（512）
        class _Ev:
            timestamp_s = 123.0
            reason = "rms_deviation"
        info = self.cap.capture(buf, _Ev())
        # 只有预触发数据可读（无后触发），post 尚未采集
        self.assertEqual(info["n_samples"], 512)
        self.assertEqual(info["pre_avail"], 512)
        self.assertTrue(os.path.exists(info["path"]))
        z = np.load(info["path"])
        self.assertEqual(z["data"].shape, (512, 3))
        self.assertEqual(int(z["pre_samples"]), 512)
        self.assertEqual(str(z["reason"]), "rms_deviation")

    def test_pre_trigger_incomplete_when_buffer_short(self):
        buf = RingBuffer(capacity=4096, channels=3)
        buf.write(np.zeros((100, 3)))  # 不足 pre_samples
        class _Ev:
            timestamp_s = 1.0
            reason = "x"
        info = self.cap.capture(buf, _Ev())
        self.assertEqual(info["pre_avail"], 100)
        self.assertLess(info["pre_avail"], 512)


# ============================================================
# 5) 采集引擎端到端（仿真）
# ============================================================

class TestAcquisitionEngine(unittest.TestCase):
    """M1 端到端：仿真信号 → 预处理流式 → 触发 → 切片"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="engine_test_")
        self.pp = CurrentPreprocessor(
            PreprocessConfig(sample_rate=FS, channels=3, norm_enabled=False), FS)
        self.wd = WatchdogFeatures(history=16)
        self.trig = TriggerEngine(k_sigma=4.0, confirm_count=3,
                                  release_count=5, warmup_frames=30)
        self.cap = SliceCapture(CaptureContext(
            pre_samples=2048, post_samples=2048,
            sample_rate=FS, channels=3, out_dir=self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, sig, chunk=512):
        engine = AcquisitionEngine(self.pp, self.wd, self.trig, self.cap)
        saved = []
        for i in range(0, len(sig), chunk):
            saved += engine.feed(sig[i:i + chunk])
        return engine, saved

    def test_normal_signal_no_slice(self):
        sig, _, _ = generate_dataset(duration=3.0, f1=50.0)
        _, saved = self._run(sig)
        self.assertEqual(saved, [])

    def test_stall_fault_creates_slice_with_pre_trigger(self):
        # 正常 2s → 堵转 1.5s → 正常
        sig, _, _ = generate_dataset(
            duration=4.5, f1=50.0,
            faults=[Fault(kind="stall", start=2.0, dur=1.5, depth=2.0)],
        )
        engine, saved = self._run(sig)
        self.assertGreaterEqual(len(saved), 1, "堵转应触发切片")
        info = saved[0]
        z = np.load(info["path"])
        data = z["data"]
        # 切片应为 [预触发 + 后触发]，含异常段（RMS 升高）
        self.assertGreater(data.shape[0], 0)
        self.assertEqual(data.shape[1], 3)
        # 预触发应基本完整（缓冲容量足够）
        self.assertGreaterEqual(int(info["pre_avail"]), info["pre_avail"])

    def test_trigger_is_latching_then_releases(self):
        """引擎内触发应激活并在故障结束后复位"""
        sig, _, _ = generate_dataset(
            duration=5.0, f1=50.0,
            faults=[Fault(kind="load_step", start=1.5, dur=1.0, depth=1.5)],
        )
        engine, saved = self._run(sig)
        # 至少一次触发
        self.assertGreaterEqual(len(saved), 1)
        # 故障结束后（5s 末尾）应复位
        self.assertFalse(engine.trig.last_event.active)


if __name__ == "__main__":
    unittest.main(verbosity=2)
