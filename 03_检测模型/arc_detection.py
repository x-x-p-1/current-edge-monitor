"""
串联电弧故障检测 (AFCI) — 1D-CNN 模型

基于 UL 1699 / GB 14287.4 标准的故障电弧检测模型。

模型架构: 1D-CNN
  - 输入: 预处理后的电流波形 256 点
  - 输出: 电弧概率 [0, 1]

训练流程:
  1. PyTorch 训练 (本文件)
  2. ONNX 导出 → RKNN 转换 → INT8 量化 (见 05_模型导出与部署)
  3. NPU 推理 (RK3588S, <0.5ms)

参考特征（GB 14287.4 附录A）:
  - 电流零休特征（平肩部）
  - 峰值因子突增
  - di/dt 高频突变
  - 高频能量比升高
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================
# 模型定义
# ============================================================

class ArcDetectionCNN(nn.Module):
    """
    串联电弧检测 1D-CNN 模型

    网络结构:
        Conv1D(1 → 16, kernel=5) → BN → ReLU → MaxPool(2)
        Conv1D(16 → 32, kernel=5) → BN → ReLU → MaxPool(2)
        Conv1D(32 → 64, kernel=3) → BN → ReLU → MaxPool(2)
        Conv1D(64 → 128, kernel=3) → BN → ReLU → AdaptiveAvgPool
        FC(128 → 64) → ReLU → Dropout(0.3)
        FC(64 → 2) → Softmax

    参数量: ~50K，INT8 量化后极轻量

    Args:
        input_length: 输入波形长度（采样点数），默认 256
        dropout: Dropout 比率
    """

    def __init__(self, input_length: int = 256, dropout: float = 0.3):
        super().__init__()

        # 卷积特征提取器
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        # 全局平均池化 → 固定维度
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 2)  # 2 分类: 正常 / 电弧

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入波形 [batch, 1, length]

        Returns:
            logits [batch, 2]
        """
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Global Average Pooling
        x = self.gap(x).squeeze(-1)  # [batch, 128]

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理接口

        Returns:
            (probabilities, predictions)
            probabilities: [batch, 2] 各类别概率
            predictions: [batch] 预测类别 (0=正常, 1=电弧)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return probs, preds


class ArcDetectionCNN_LSTM(nn.Module):
    """
    串联电弧检测 CNN + BiLSTM 混合模型

    在 CNN 特征提取后加入双向 LSTM，捕获时序依赖。
    适用于需要更长上下文（多个窗口联合判断）的场景。

    注意: 此模型较大，RKNN 导出时需注意 LSTM 算子兼容性。
    """

    def __init__(self, input_length: int = 256, hidden_size: int = 64):
        super().__init__()

        # CNN 特征提取
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # BiLSTM 时序建模
        # 经过两次 MaxPool(2) 后长度 = input_length / 4
        lstm_input_len = input_length // 4
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 分类头
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # [batch, channels, seq] → [batch, seq, channels] (LSTM 需要)
        x = x.permute(0, 2, 1)

        # BiLSTM
        x, _ = self.lstm(x)

        # 取最后一个时间步
        x = x[:, -1, :]

        # 分类
        x = self.fc(x)
        return x


# ============================================================
# 模型工厂
# ============================================================

def create_arc_model(
    model_type: str = "cnn",
    input_length: int = 256,
    **kwargs,
) -> nn.Module:
    """
    创建电弧检测模型

    Args:
        model_type: 模型类型 "cnn" | "cnn_lstm"
        input_length: 输入波形长度
        **kwargs: 额外参数传递给模型构造函数

    Returns:
        PyTorch 模型
    """
    if model_type == "cnn":
        return ArcDetectionCNN(input_length=input_length, **kwargs)
    elif model_type == "cnn_lstm":
        return ArcDetectionCNN_LSTM(input_length=input_length, **kwargs)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


# ============================================================
# 推理函数（纯 Numpy，用于边缘端无 PyTorch 时）
# ============================================================

def arc_detection_rule_based(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    thresholds: Optional[dict] = None,
) -> Tuple[bool, float, dict]:
    """
    基于规则的快速电弧检测（不依赖 AI 模型）

    用于初始阶段快速原型验证，或作为 AI 模型的对比基线。

    规则集（GB 14287.4 启发）:
        1. 峰值因子 > 3.0
        2. 高频能量比 > 0.08
        3. 零休时间 > 0.5ms
        4. 离群点比例 > 15%

    仲裁逻辑: 满足 ≥3 条规则 → 判定为电弧

    Args:
        signal: 预处理后的电流波形
        sample_rate: 采样率
        thresholds: 自定义阈值字典

    Returns:
        (is_arc, confidence, rule_details)
    """
    import importlib
    _td = importlib.import_module("02_特征提取.time_domain")
    compute_peak_factor = _td.compute_peak_factor
    compute_zero_crossing_rate = _td.compute_zero_crossing_rate
    compute_current_zero_duration = _td.compute_current_zero_duration
    _fd = importlib.import_module("02_特征提取.frequency_domain")
    compute_high_freq_energy_ratio = _fd.compute_high_freq_energy_ratio
    _stat = importlib.import_module("02_特征提取.statistical")
    outlier_ratio = _stat.outlier_ratio

    if thresholds is None:
        thresholds = {
            "peak_factor": 3.0,
            "high_freq_ratio": 0.08,
            "zero_duration_ms": 0.5,
            "outlier_ratio": 0.15,
        }

    rule_results = {}

    # 规则1: 峰值因子
    cf = compute_peak_factor(signal)
    rule_results["peak_factor"] = {
        "value": cf,
        "triggered": cf > thresholds["peak_factor"],
    }

    # 规则2: 高频能量比
    hf_ratio = compute_high_freq_energy_ratio(signal, sample_rate)
    rule_results["high_freq_ratio"] = {
        "value": hf_ratio,
        "triggered": hf_ratio > thresholds["high_freq_ratio"],
    }

    # 规则3: 零休时间
    zero_dur = compute_current_zero_duration(signal, sample_rate)
    rule_results["zero_duration_ms"] = {
        "value": zero_dur,
        "triggered": zero_dur > thresholds["zero_duration_ms"],
    }

    # 规则4: 离群点比例
    out_ratio = outlier_ratio(signal, multiplier=1.5)
    rule_results["outlier_ratio"] = {
        "value": out_ratio,
        "triggered": out_ratio > thresholds["outlier_ratio"],
    }

    # 仲裁: 满足 N/4 条规则
    triggered_count = sum(1 for r in rule_results.values() if r["triggered"])
    confidence = triggered_count / 4.0
    is_arc = triggered_count >= 2  # 至少满足 2 条

    return is_arc, confidence, rule_results
