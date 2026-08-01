"""
测试与验证 — 检测模型模块

验证 AI 模型的前向传播和输出格式。
"""

import sys
import os
import importlib
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

_arc = importlib.import_module("03_检测模型.arc_detection")
ArcDetectionCNN = _arc.ArcDetectionCNN
ArcDetectionCNN_LSTM = _arc.ArcDetectionCNN_LSTM
create_arc_model = _arc.create_arc_model
arc_detection_rule_based = _arc.arc_detection_rule_based

_ae = importlib.import_module("03_检测模型.anomaly_detection")
CurrentAutoEncoder = _ae.CurrentAutoEncoder
AnomalyDetector = _ae.AnomalyDetector
create_anomaly_detector = _ae.create_anomaly_detector

_ld = importlib.import_module("03_检测模型.load_identification")
LoadClassifier1DResNet = _ld.LoadClassifier1DResNet
LOAD_CLASSES = _ld.LOAD_CLASSES

_pq = importlib.import_module("03_检测模型.power_quality")
analyze_power_quality = _pq.analyze_power_quality
PowerQualityReport = _pq.PowerQualityReport
PowerQualityStatus = _pq.PowerQualityStatus
estimate_frequency_interpolated_fft = _pq.estimate_frequency_interpolated_fft
compute_rms_half_cycle = _pq.compute_rms_half_cycle


class TestArcDetectionCNN(unittest.TestCase):
    """电弧检测 CNN 模型测试"""

    def setUp(self):
        self.batch_size = 4
        self.input_length = 256
        self.model = ArcDetectionCNN(input_length=self.input_length)

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        x = torch.randn(self.batch_size, 1, self.input_length)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 2))

    def test_predict(self):
        """测试推理接口"""
        x = torch.randn(self.batch_size, 1, self.input_length)
        probs, preds = self.model.predict(x)
        self.assertEqual(probs.shape, (self.batch_size, 2))
        self.assertEqual(preds.shape, (self.batch_size,))
        # 概率之和应为 1
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(self.batch_size)))

    def test_model_factory(self):
        """测试模型工厂"""
        model = create_arc_model("cnn", input_length=256)
        self.assertIsInstance(model, ArcDetectionCNN)

        model_lstm = create_arc_model("cnn_lstm", input_length=256)
        self.assertIsInstance(model_lstm, ArcDetectionCNN_LSTM)

    def test_rule_based_detection(self):
        """测试规则基线检测"""
        sample_rate = 50000.0
        t = np.arange(256) / sample_rate
        # 模拟正常正弦波
        normal = np.sin(2 * np.pi * 50 * t)
        is_arc, conf, details = arc_detection_rule_based(normal, sample_rate)
        self.assertIsInstance(is_arc, bool)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestAutoEncoder(unittest.TestCase):
    """自编码器模型测试"""

    def setUp(self):
        self.input_length = 256
        self.model = CurrentAutoEncoder(input_length=self.input_length)

    def test_encode_decode_shape(self):
        """测试编码-解码输出形状"""
        x = torch.randn(1, 1, self.input_length)
        z = self.model.encode(x)
        self.assertEqual(z.shape[0], 1)

        recon = self.model.decode(z)
        # 转置卷积可能有 ±1 偏差
        self.assertAlmostEqual(recon.shape[-1], self.input_length, delta=2)

    def test_reconstruction_error(self):
        """测试重构误差计算"""
        x = torch.randn(4, 1, self.input_length)
        error = self.model.reconstruction_error(x)
        self.assertGreater(error.item(), 0.0)


class TestLoadClassifier(unittest.TestCase):
    """负载分类器测试"""

    def setUp(self):
        self.model = LoadClassifier1DResNet(num_classes=8)

    def test_forward_shape(self):
        """测试输出形状"""
        x = torch.randn(4, 1, 256)
        output = self.model(x)
        self.assertEqual(output.shape, (4, 8))

    def test_predict(self):
        """测试推理接口"""
        x = torch.randn(4, 1, 256)
        probs, preds = self.model.predict(x)
        self.assertEqual(probs.shape, (4, 8))
        self.assertEqual(preds.shape, (4,))

    def test_class_names(self):
        """测试类别名称"""
        self.assertEqual(len(LOAD_CLASSES), 8)


class TestPowerQuality(unittest.TestCase):
    """电能质量分析测试"""

    def setUp(self):
        self.sample_rate = 50000.0
        self.t = np.arange(256) / self.sample_rate

    def test_frequency_estimation(self):
        """测试频率估计"""
        signal = np.sin(2 * np.pi * 50 * self.t)
        freq = estimate_frequency_interpolated_fft(signal, self.sample_rate)
        self.assertAlmostEqual(freq, 50.0, delta=1.0)

    def test_half_cycle_rms(self):
        """测试半周期 RMS — 信号需 ≥ 半周期采样数（50k/50Hz 下为 500 点）"""
        t_long = np.arange(2000) / self.sample_rate  # 4 个工频周期
        signal = np.sin(2 * np.pi * 50 * t_long)
        rms_seq = compute_rms_half_cycle(signal, self.sample_rate, 50.0)
        self.assertGreater(len(rms_seq), 0)

    def test_power_quality_report(self):
        """测试综合电能质量报告"""
        signal = np.sin(2 * np.pi * 50 * self.t)
        report = analyze_power_quality(signal, self.sample_rate, 50.0)
        self.assertIsInstance(report, PowerQualityReport)
        self.assertIsInstance(report.to_dict(), dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
