"""
测试与验证 — 信号预处理模块

验证滤波器、归一化、相位对齐等功能。
"""

import sys
import os
import importlib
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Python 模块名不能以数字开头，使用 importlib 导入
_filters = importlib.import_module("01_信号预处理.filters")
remove_dc_offset = _filters.remove_dc_offset
bandpass_filter = _filters.bandpass_filter
butter_bandpass_coefficients = _filters.butter_bandpass_coefficients
moving_average = _filters.moving_average
savitzky_golay_smooth = _filters.savitzky_golay_smooth
SlowBaselineRemover = _filters.SlowBaselineRemover
StreamingBandpass = _filters.StreamingBandpass


def _rms_at_freq(x: np.ndarray, fs: float, f: float) -> float:
    """整周期窗下单频有效值（抗泄漏，用于基波保留率测量）"""
    n = len(x)
    k = round(f * n / fs)
    if k < 1 or k >= n // 2:
        return 0.0
    return np.abs(np.fft.rfft(x)[k]) / n * np.sqrt(2)

_norm = importlib.import_module("01_信号预处理.normalization")
normalize_signal = _norm.normalize_signal
NormalizationMethod = _norm.NormalizationMethod
compute_normalization_params = _norm.compute_normalization_params
apply_normalization_params = _norm.apply_normalization_params

_align = importlib.import_module("01_信号预处理.alignment")
find_zero_crossings = _align.find_zero_crossings
align_to_zero_crossing = _align.align_to_zero_crossing
extract_full_cycle = _align.extract_full_cycle

_preproc = importlib.import_module("01_信号预处理.preprocess")
CurrentPreprocessor = _preproc.CurrentPreprocessor
PreprocessConfig = _preproc.PreprocessConfig


class TestFilters(unittest.TestCase):
    """滤波器测试"""

    def setUp(self):
        self.sample_rate = 50000.0
        self.n = 256
        self.t = np.arange(self.n) / self.sample_rate

    def test_remove_dc_offset(self):
        """测试 DC 偏置去除"""
        # 生成带 DC 偏置的正弦信号
        signal = 2.5 + np.sin(2 * np.pi * 50 * self.t)

        result = remove_dc_offset(signal, window=50)

        # 均值应接近 0
        self.assertLess(abs(np.mean(result)), 0.1)

    def test_bandpass_filter(self):
        """测试带通滤波器"""
        # 生成含多频率分量的信号
        signal = (
            np.sin(2 * np.pi * 50 * self.t)     # 50Hz 基波（应保留）
            + 0.5 * np.sin(2 * np.pi * 150 * self.t)  # 150Hz 3次谐波（应保留）
            + 0.1 * np.sin(2 * np.pi * 3000 * self.t) # 3kHz（应滤除）
        )

        b, a = butter_bandpass_coefficients(45, 2500, self.sample_rate)
        result = bandpass_filter(signal, b, a)

        # 滤波后高频分量应被衰减
        self.assertLess(np.std(result), np.std(signal))

    def test_savitzky_golay_smooth(self):
        """测试 Savitzky-Golay 平滑"""
        signal = np.sin(2 * np.pi * 50 * self.t) + 0.1 * np.random.randn(self.n)

        smoothed = savitzky_golay_smooth(signal, window_length=5, polyorder=3)

        self.assertEqual(len(smoothed), len(signal))


class TestDCBaselineV21(unittest.TestCase):
    """v2.1：慢基线去直流 — 基波保留（修复滑动均值窗砍基波 bug，TODO G 项）"""

    def setUp(self):
        self.fs = 16000.0
        self.f1 = 50.0
        n = int(self.fs * 4.0)
        t = np.arange(n) / self.fs
        np.random.seed(0)
        self.sig = (2.5 + 1.0 * np.sin(2 * np.pi * self.f1 * t)
                    + 0.2 * np.sin(2 * np.pi * 3 * self.f1 * t)
                    + 0.01 * np.random.randn(n))

    def _fund_gain(self, sig_out: np.ndarray, ref: np.ndarray) -> float:
        g_in = _rms_at_freq(ref, self.fs, self.f1)
        g_out = _rms_at_freq(sig_out, self.fs, self.f1)
        return g_out / max(g_in, 1e-12)

    def test_sliding_mean_small_window_kills_fundamental(self):
        """哨兵：@16k window=100 的滑动均值确实会砍基波（bug 复现）"""
        out = remove_dc_offset(self.sig, window=100)
        self.assertLess(self._fund_gain(out, self.sig), 0.4)  # 砍掉 >60%

    def test_highpass_preserves_fundamental(self):
        """慢基线高通去直流保留基波 ~100%，且 DC 被抑制"""
        out = remove_dc_offset(self.sig, method="highpass", cutoff_hz=0.5, fs=self.fs)
        self.assertGreater(self._fund_gain(out, self.sig), 0.98)
        self.assertLess(abs(np.mean(out)), 0.1)

    def test_sliding_mean_full_cycle_preserves(self):
        """滑动均值窗口 ≥ 1 工频周期时也保留基波（文档要求）"""
        out = remove_dc_offset(self.sig, window=320)  # 320 点 = 1 周期 @16k/50Hz
        self.assertGreater(self._fund_gain(out, self.sig), 0.98)

    def test_streaming_pipeline_preserves_fundamental(self):
        """流式管线（慢基线 + 带通）重建后基波保留 >95%"""
        cfg = PreprocessConfig(sample_rate=self.fs, channels=1, norm_enabled=False)
        pp = CurrentPreprocessor(cfg, sample_rate=self.fs)
        frames = []
        for i in range(0, len(self.sig), 64):
            frames += pp.process_streaming(self.sig[i:i + 64])
        recon = np.vstack([f[:cfg.stride] for f in frames])[:, 0]
        seg = recon[-int(2.0 * self.fs):]
        ref = self.sig[-int(2.0 * self.fs):]
        self.assertGreater(self._fund_gain(seg, ref), 0.95)

    def test_streaming_matches_ideal_filtering(self):
        """流式重建 ≈ 整段因果滤波（r > 0.999）：状态延续无帧边界瞬态"""
        from scipy import signal as sgl
        cfg = PreprocessConfig(sample_rate=self.fs, channels=1, norm_enabled=False)
        pp = CurrentPreprocessor(cfg, sample_rate=self.fs)
        frames = []
        for i in range(0, len(self.sig), 64):
            frames += pp.process_streaming(self.sig[i:i + 64])
        recon = np.vstack([f[:cfg.stride] for f in frames])[:, 0]

        sos_high = sgl.butter(1, cfg.dc_cutoff_hz / (0.5 * self.fs), btype="high", output="sos")
        sos_bp = sgl.butter(
            cfg.filter_order,
            [cfg.filter_lowcut / (0.5 * self.fs), cfg.filter_highcut / (0.5 * self.fs)],
            btype="band", output="sos",
        )
        ideal = sgl.sosfilt(sos_bp, sgl.sosfilt(sos_high, self.sig))
        m = min(len(recon), len(ideal))
        tail = np.arange(m - int(self.fs), m)
        r = np.corrcoef(recon[tail], ideal[tail])[0, 1]
        self.assertGreater(r, 0.999)

    def test_slow_baseline_streaming_state_continuity(self):
        """SlowBaselineRemover 状态延续：不同喂入粒度输出逐点一致"""
        def run(chunk):
            dc = SlowBaselineRemover(cutoff_hz=0.5, fs=self.fs, channels=1)
            outs = [dc(self.sig[i:i + chunk]) for i in range(0, len(self.sig), chunk)]
            return np.concatenate(outs)
        a64 = run(64)
        a256 = run(256)
        # 粒度只影响起始瞬态（zi 用首块均值初始化），稳态应一致；
        # 容忍瞬态残差，但检测结构性错位（未延续状态 → 差异 O(1)）
        np.testing.assert_allclose(a64[-int(self.fs):], a256[-int(self.fs):], atol=1e-3)


class TestNormalization(unittest.TestCase):
    """归一化测试"""

    def setUp(self):
        np.random.seed(42)
        self.signal = np.random.randn(256) * 2 + 5  # μ=5, σ=2

    def test_zscore_normalization(self):
        """测试 Z-Score 归一化"""
        result = normalize_signal(self.signal, NormalizationMethod.ZSCORE)
        self.assertAlmostEqual(np.mean(result), 0.0, places=5)
        self.assertAlmostEqual(np.std(result), 1.0, places=5)

    def test_minmax_normalization(self):
        """测试 Min-Max 归一化"""
        result = normalize_signal(self.signal, NormalizationMethod.MINMAX)
        self.assertAlmostEqual(np.min(result), 0.0, places=5)
        self.assertAlmostEqual(np.max(result), 1.0, places=5)

    def test_params_consistency(self):
        """测试参数保存/应用一致性"""
        params = compute_normalization_params(self.signal, NormalizationMethod.ZSCORE)
        result1 = normalize_signal(self.signal, NormalizationMethod.ZSCORE)
        result2 = apply_normalization_params(self.signal, params, NormalizationMethod.ZSCORE)
        np.testing.assert_array_almost_equal(result1, result2)


class TestAlignment(unittest.TestCase):
    """相位对齐测试"""

    def setUp(self):
        self.sample_rate = 50000.0
        self.t = np.arange(500) / self.sample_rate

    def test_find_zero_crossings(self):
        """测试过零点检测 — 使用非整数周期避免采样点恰好在零点"""
        # 49Hz @ 50kHz = 1020.4 点/周期，不会恰好在零点
        t_long = np.arange(2000) / self.sample_rate
        signal = np.sin(2 * np.pi * 49 * t_long)

        zero_crossings = find_zero_crossings(signal, direction="positive", tolerance=0.001)
        self.assertGreater(len(zero_crossings), 0)

    def test_align_to_zero_crossing(self):
        """测试相位对齐"""
        signal = np.sin(2 * np.pi * 50 * self.t)

        aligned = align_to_zero_crossing(signal, target_length=128)
        self.assertEqual(len(aligned), 128)
        # 对齐后第一个采样点应接近 0（正向上坡）
        self.assertAlmostEqual(aligned[0], 0.0, delta=0.05)


class TestPreprocessPipeline(unittest.TestCase):
    """预处理管线集成测试"""

    def setUp(self):
        self.sample_rate = 50000.0
        self.t = np.arange(256) / self.sample_rate

    def test_full_pipeline(self):
        """测试完整预处理管线"""
        # 模拟带噪声、DC偏置的电流信号
        signal = (
            1.2                                 # DC 偏置
            + np.sin(2 * np.pi * 50 * self.t)   # 基波
            + 0.3 * np.sin(2 * np.pi * 150 * self.t)  # 3次谐波
            + 0.05 * np.random.randn(256)        # 噪声
        )

        config = PreprocessConfig(
            dc_removal_enabled=True,
            filter_enabled=True,
            norm_enabled=True,
            norm_method="zscore",
        )
        preprocessor = CurrentPreprocessor(config, self.sample_rate)
        result = preprocessor.process(signal)

        # 输出应该是干净的归一化信号
        self.assertEqual(len(result), 256)
        self.assertLess(abs(np.mean(result)), 0.2)  # 均值接近0
        # 标准差应该在合理范围（归一化后 ≈ 1.0）
        self.assertGreater(np.std(result), 0.5)
        self.assertLess(np.std(result), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
