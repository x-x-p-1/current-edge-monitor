"""
无监督异常检测 — AutoEncoder 模型

当缺乏标注数据时，使用自编码器学习正常电流波形的分布，
通过重构误差来检测异常。

原理（借鉴西门子 DKW 的自适应基线思想）:
  1. 训练阶段: 仅用正常负载的电流波形训练 AutoEncoder
  2. 推理阶段: 对输入波形编码→解码，计算重构误差
  3. 异常判定: 重构误差 >> 训练集基准误差 → 异常

优点:
  - 不需要故障数据标注
  - 对未知类型的异常也有检测能力
  - 可以作为电弧 CNN 等监督模型的补充

参考:
  - 西门子 CMS 的自适应基线校准机制
  - 工业预测性维护中的无监督异常检测范式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================
# 模型定义
# ============================================================

class CurrentAutoEncoder(nn.Module):
    """
    电流信号自编码器

    编码器: 1D信号 → 压缩潜在表示
    解码器: 潜在表示 → 重构信号

    潜在空间维度: 32（在 256 点输入下压缩 8 倍）
    """

    def __init__(
        self,
        input_length: int = 256,
        latent_dim: int = 32,
    ):
        super().__init__()

        # ── 编码器 ──
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 计算编码后长度
        enc_length = input_length // 16  # 4 次 stride=2

        # 展平 → 潜在空间
        self.fc_enc = nn.Linear(128 * enc_length, latent_dim)

        # ── 解码器 ──
        self.fc_dec = nn.Linear(latent_dim, 128 * enc_length)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        self.enc_length = enc_length

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_enc(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        z = self.fc_dec(z)
        z = z.view(z.size(0), 128, self.enc_length)
        z = self.decoder(z)
        # 裁剪到原始长度（因转置卷积可能有 ±1 偏差）
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整的前向传播: 编码→解码"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        计算重构误差（MSE）

        Args:
            x: 输入波形 [batch, 1, length]
            reduction: "mean" | "none"

        Returns:
            重构误差
        """
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            # 对齐长度
            min_len = min(x.shape[-1], x_recon.shape[-1])
            x = x[..., :min_len]
            x_recon = x_recon[..., :min_len]

            error = F.mse_loss(x_recon, x, reduction=reduction)

        return error


# ============================================================
# 异常检测逻辑
# ============================================================

class AnomalyDetector:
    """
    基于 AutoEncoder 的异常检测器

    工作流程:
      1. 训练阶段: 收集正常样本的统计信息
      2. 推理阶段: 计算重构误差 → 与基线比较 → 异常判定
    """

    def __init__(
        self,
        model: CurrentAutoEncoder,
        baseline_error_mean: float = 0.0,
        baseline_error_std: float = 1.0,
        threshold_multiplier: float = 3.0,
    ):
        self.model = model
        self.baseline_mean = baseline_error_mean
        self.baseline_std = baseline_error_std
        self.threshold_multiplier = threshold_multiplier

    def calibrate(
        self,
        normal_samples: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        基线校准（类似西门子 DKW 的基线学习）

        用大量正常样本计算重构误差的分布，
        以此作为"健康基准"。

        Args:
            normal_samples: 正常波形样本 [N, 1, length]

        Returns:
            (mean_recon_error, std_recon_error)
        """
        errors = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0, len(normal_samples), 64):
                batch = normal_samples[i:i + 64]
                error = self.model.reconstruction_error(batch, reduction="none")
                errors.extend(error.cpu().numpy())

        errors = np.array(errors)
        self.baseline_mean = float(np.mean(errors))
        self.baseline_std = float(np.std(errors))

        return self.baseline_mean, self.baseline_std

    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        异常检测推理

        Args:
            x: 输入波形 [batch, 1, length]

        Returns:
            (anomaly_scores, is_anomaly)
            anomaly_scores: Z-Score 形式 (误差偏离基线多少σ)
            is_anomaly: 是否判定为异常
        """
        error = self.model.reconstruction_error(x, reduction="none")
        error_np = error.cpu().numpy()

        # Z-Score 异常分数
        if self.baseline_std > 1e-8:
            anomaly_scores = (error_np - self.baseline_mean) / self.baseline_std
        else:
            anomaly_scores = error_np - self.baseline_mean

        # 异常判定
        is_anomaly = anomaly_scores > self.threshold_multiplier

        return anomaly_scores, is_anomaly


# ============================================================
# 模型工厂
# ============================================================

def create_anomaly_detector(
    input_length: int = 256,
    latent_dim: int = 32,
    pretrained_path: Optional[str] = None,
) -> AnomalyDetector:
    """
    创建异常检测器

    Args:
        input_length: 输入波形长度
        latent_dim: 潜在空间维度
        pretrained_path: 预训练权重路径

    Returns:
        AnomalyDetector 实例
    """
    model = CurrentAutoEncoder(input_length=input_length, latent_dim=latent_dim)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

    return AnomalyDetector(model)
