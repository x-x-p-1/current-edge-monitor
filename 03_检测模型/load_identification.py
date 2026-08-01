"""
负载类型识别 (NILM) — 1D-ResNet 模型

基于稳态电流波形特征识别接入的负载类型。

负载类别:
  0: 阻性负载 (电热器、白炽灯)
  1: 感性负载 (电机、变压器)
  2: 容性负载 (电容补偿柜)
  3: 开关电源 (LED、电脑、充电器)
  4: 变频器驱动
  5: 整流电路
  6: 混合负载
  7: 未知/异常

参考:
  - 西门子 LGF 库的统计特征分类思想
  - IEC 61000-3-2 谐波发射限值（不同负载谐波特征不同）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ============================================================
# 残差块
# ============================================================

class ResidualBlock1D(nn.Module):
    """1D 残差模块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


# ============================================================
# 1D ResNet
# ============================================================

class LoadClassifier1DResNet(nn.Module):
    """
    负载类型识别 1D-ResNet

    网络结构:
        Conv1D(1 → 32, k=7, s=2)
        MaxPool(3, s=2)
        ResBlock(32 → 64)  ×2
        ResBlock(64 → 128) ×2
        ResBlock(128 → 256) ×2
        GAP → FC(256 → num_classes)

    参数量: ~200K
    """

    def __init__(
        self,
        num_classes: int = 8,
        input_length: int = 256,
    ):
        super().__init__()

        # 初始卷积
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # 分类头
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理并返回概率和预测类别"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return probs, preds


# ============================================================
# 默认负载类别定义
# ============================================================

LOAD_CLASSES = [
    "阻性负载",
    "感性负载",
    "容性负载",
    "开关电源",
    "变频器驱动",
    "整流电路",
    "混合负载",
    "未知/异常",
]

LOAD_CLASSES_EN = [
    "resistive",
    "inductive",
    "capacitive",
    "switching_psu",
    "vfd_drive",
    "rectifier",
    "mixed",
    "unknown",
]


def create_load_classifier(
    num_classes: int = 8,
    pretrained_path: Optional[str] = None,
) -> LoadClassifier1DResNet:
    """
    创建负载分类器

    Args:
        num_classes: 类别数
        pretrained_path: 预训练权重

    Returns:
        LoadClassifier1DResNet
    """
    model = LoadClassifier1DResNet(num_classes=num_classes)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

    return model
