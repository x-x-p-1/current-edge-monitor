"""
时频域特征提取模块

当单纯的时域或频域特征不足以区分正常和故障状态时，
需要使用时频联合分析方法。

方法:
  1. 离散小波变换 (DWT) — 多分辨率分析
  2. 短时傅里叶变换 (STFT) — 时变频谱
  3. 离散小波包能量 — 频段子带能量

在电流检测中的应用:
  - 电弧是高度非平稳信号，传统 FFT 无法捕捉其时变特性
  - 小波变换能同时定位时间和频率上的突变
  - DWT 细节系数 D1-D4 可捕捉不同频段的瞬态特征

借鉴西门子希尔伯特包络解调思想:
  通过时频域解耦，分离不同物理来源的信号分量。
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# ============================================================
# 离散小波变换 (DWT)
# ============================================================

def dwt_decompose(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> List[np.ndarray]:
    """
    离散小波多级分解

    将信号分解为:
      - cA_L: 第 L 级近似系数（低频概貌）
      - cD_1, cD_2, ..., cD_L: 各层细节系数（高频细节）

    频段划分（以 50kSPS, 4 级 db4 为例）:
      - 原始: 0 ~ 25 kHz (Nyquist)
      - cD1: 12.5 ~ 25 kHz  (超高频)
      - cD2: 6.25 ~ 12.5 kHz (高频)
      - cD3: 3.125 ~ 6.25 kHz (中高频 — 电弧特征区)
      - cD4: 1.5625 ~ 3.125 kHz (中频)
      - cA4: 0 ~ 1.5625 kHz (低频 + 基波)

    Args:
        signal: 输入信号
        wavelet: 小波基名称 (db4, sym4, coif4 等)
        level: 分解层数

    Returns:
        [cA_L, cD_1, cD_2, ..., cD_L]
        注意: 返回顺序是从最深层近似到最浅层细节
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "需要安装 PyWavelets: pip install pywt"
        )

    signal = np.asarray(signal, dtype=np.float64)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs  # [cA_n, cD_n, cD_n-1, ..., cD_1]


def dwt_detail_energy(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> Dict[str, float]:
    """
    计算各级小波细节系数的能量

    能量 = Σ(coeff²)

    电弧检测意义:
        电弧信号的 cD3 和 cD2 细节系数能量会显著高于正常负载，
        因为这些频段(1.5kHz~12.5kHz)正好对应电弧的高频噪声。

    Args:
        signal: 输入信号
        wavelet: 小波基
        level: 分解层数

    Returns:
        {"cD1_energy": ..., "cD2_energy": ..., ...}
    """
    coeffs = dwt_decompose(signal, wavelet, level)
    # coeffs = [cA_level, cD_level, cD_level-1, ..., cD_1]

    results = {}
    results["cA_energy"] = float(np.sum(np.square(coeffs[0])))

    for i, coef in enumerate(coeffs[1:], start=1):
        results[f"cD{i}_energy"] = float(np.sum(np.square(coef)))

    return results


def dwt_detail_entropy(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> Dict[str, float]:
    """
    计算各级小波细节系数的香农熵

    熵 = -Σ(p_i * log(p_i))，其中 p_i = coeff_i² / Σcoeff²

    熵越高表示该频段的信号越"无序"。
    电弧信号的 cD2/cD3 熵通常高于正常负载。

    Args:
        signal: 输入信号
        wavelet: 小波基
        level: 分解层数

    Returns:
        各级系数的熵值字典
    """
    coeffs = dwt_decompose(signal, wavelet, level)
    results = {}
    eps = 1e-12

    for i, coef in enumerate(coeffs):
        name = "cA" if i == 0 else f"cD{level - i + 1}"
        energy = np.square(coef)
        total = np.sum(energy) + eps
        prob = energy / total
        # Shannon entropy
        ent = -np.sum(prob * np.log(prob + eps))
        results[f"{name}_entropy"] = float(ent)

    return results


def dwt_feature_vector(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> np.ndarray:
    """
    提取 DWT 特征向量（能量 + 熵）

    用于作为 ML 模型的输入特征。

    Returns:
        特征向量，shape (2*level + 2,)
    """
    energy = dwt_detail_energy(signal, wavelet, level)
    entropy = dwt_detail_entropy(signal, wavelet, level)

    # 按固定顺序拼接
    feat_list = []
    feat_list.append(energy.get("cA_energy", 0.0))
    for i in range(1, level + 1):
        feat_list.append(energy.get(f"cD{i}_energy", 0.0))
    feat_list.append(entropy.get("cA_entropy", 0.0))
    for i in range(1, level + 1):
        feat_list.append(entropy.get(f"cD{i}_entropy", 0.0))

    return np.array(feat_list, dtype=np.float32)


# ============================================================
# 短时傅里叶变换 (STFT)
# ============================================================

def compute_stft_spectrogram(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nperseg: int = 64,
    noverlap: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算短时傅里叶变换谱图

    输出一个时间×频率的二维矩阵，可以直接作为 2D-CNN 的输入。

    Args:
        signal: 输入信号
        sample_rate: 采样率
        nperseg: 每个 STFT 段的采样点数
        noverlap: 段间重叠点数

    Returns:
        (frequencies, times, spectrogram)
        spectrogram shape: (n_freq, n_time)
    """
    f, t, Zxx = scipy_signal.stft(
        signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
    )
    return f, t, np.abs(Zxx)


def stft_energy_bands(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nperseg: int = 64,
    noverlap: int = 32,
    bands: Optional[List[Tuple[float, float]]] = None,
) -> List[np.ndarray]:
    """
    计算 STFT 各频段随时间变化的能量

    Args:
        signal: 输入信号
        sample_rate: 采样率
        nperseg: 段长度
        noverlap: 重叠
        bands: 频段列表

    Returns:
        各频段的时间序列能量
    """
    f, t, Zxx = compute_stft_spectrogram(signal, sample_rate, nperseg, noverlap)

    if bands is None:
        bands = [
            (45, 55),
            (100, 350),
            (500, 2000),
            (2000, 5000),
        ]

    band_energies = []
    for low, high in bands:
        mask = (f >= low) & (f <= high)
        band_power = np.sum(np.abs(Zxx[mask, :]) ** 2, axis=0)
        band_energies.append(band_power)

    return band_energies


# ============================================================
# 希尔伯特变换 (借鉴西门子包络解调)
# ============================================================

def hilbert_envelope(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算希尔伯特包络线

    借鉴西门子 SM1281 的硬件包络解调三步法思想:
      1. 带通提纯 → 2. 希尔伯特变换 → 3. 包络提取

    在电流检测中的应用:
      - 对滤波后的电流信号做包络分析
      - 电弧电流的包络线会比正弦波更不规则
      - 包络的方差/峰峰值可作为特征

    Args:
        signal: 输入信号

    Returns:
        (analytic_signal_amplitude, instantaneous_phase)
    """
    analytic = scipy_signal.hilbert(signal)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    return envelope, phase


def envelope_statistics(signal: np.ndarray) -> Dict[str, float]:
    """
    计算包络线的统计特征

    Returns:
        dict with:
          - env_mean: 包络均值
          - env_std: 包络标准差
          - env_range: 包络峰-峰值
          - env_cv: 包络变异系数 (std/mean)
    """
    envelope, _ = hilbert_envelope(signal)
    env_mean = float(np.mean(envelope))
    env_std = float(np.std(envelope))
    env_range = float(np.max(envelope) - np.min(envelope))
    env_cv = float(env_std / env_mean) if env_mean > 1e-12 else 0.0

    return {
        "env_mean": env_mean,
        "env_std": env_std,
        "env_range": env_range,
        "env_cv": env_cv,
    }


# ============================================================
# 批量特征提取
# ============================================================

@dataclass
class TimeFrequencyFeatures:
    """时频域特征集合"""
    # DWT 特征
    dwt_energy: Dict[str, float]
    dwt_entropy: Dict[str, float]

    # 包络特征
    envelope_mean: float = 0.0
    envelope_std: float = 0.0
    envelope_range: float = 0.0
    envelope_cv: float = 0.0

    def to_dict(self) -> dict:
        result = {}
        result.update(self.dwt_energy)
        result.update(self.dwt_entropy)
        result["envelope_mean"] = self.envelope_mean
        result["envelope_std"] = self.envelope_std
        result["envelope_range"] = self.envelope_range
        result["envelope_cv"] = self.envelope_cv
        return result

    def to_array(self) -> np.ndarray:
        """转为固定长度特征向量"""
        feat_list = []
        # DWT 能量 (5个)
        for key in ["cA_energy", "cD1_energy", "cD2_energy", "cD3_energy", "cD4_energy"]:
            feat_list.append(self.dwt_energy.get(key, 0.0))
        # DWT 熵 (5个)
        for key in ["cA_entropy", "cD1_entropy", "cD2_entropy", "cD3_entropy", "cD4_entropy"]:
            feat_list.append(self.dwt_entropy.get(key, 0.0))
        # 包络特征 (4个)
        feat_list.extend([self.envelope_mean, self.envelope_std,
                         self.envelope_range, self.envelope_cv])
        return np.array(feat_list, dtype=np.float32)


def extract_time_frequency_features(
    signal: np.ndarray,
    wavelet: str = "db4",
    dwt_level: int = 4,
) -> TimeFrequencyFeatures:
    """
    从电流信号中提取所有时频域特征

    Args:
        signal: 预处理后的电流采样序列
        wavelet: 小波基
        dwt_level: 小波分解层数

    Returns:
        TimeFrequencyFeatures 对象
    """
    # DWT 能量和熵
    energy = dwt_detail_energy(signal, wavelet, dwt_level)
    entropy = dwt_detail_entropy(signal, wavelet, dwt_level)

    # 包络统计
    env_stats = envelope_statistics(signal)

    return TimeFrequencyFeatures(
        dwt_energy=energy,
        dwt_entropy=entropy,
        envelope_mean=env_stats["env_mean"],
        envelope_std=env_stats["env_std"],
        envelope_range=env_stats["env_range"],
        envelope_cv=env_stats["env_cv"],
    )
