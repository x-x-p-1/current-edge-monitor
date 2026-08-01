"""
时域特征提取模块

从电流波形时域序列中提取具有明确物理含义的标量特征。
借鉴西门子 SM1281 的三维指标体系思想（vRMS, aRMS, DKW），
迁移到电流检测领域，构建电流域的"三维+多维度"特征体系。

核心特征:
  1. RMS 有效值 — 电流幅值的整体能量度量
  2. 峰值因子 — 波形尖峰程度，电弧关键指标
  3. 波形因子 — 波形偏离正弦的程度
  4. 峭度 (Kurtosis) — 脉冲型异常敏感指标
  5. 偏度 (Skewness) — 波形不对称性
  6. 过零率 — 电弧"平肩部"导致过零减少
  7. 差分统计 — 瞬态突变检测

参考:
  - GB 14287.4-2014 附录A 电弧特征参数
  - UL 1699 电弧检测方法
  - 西门子 DKW 无量纲归一化方法论
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 特征计算函数
# ============================================================

def compute_rms(signal: np.ndarray) -> float:
    """
    计算电流有效值 (Root Mean Square)

    数学定义:
        I_rms = sqrt( (1/N) * Σ i_k² )

    物理意义:
        RMS 直接对应电流的热效应——这是评估负载大小的最基本指标。

    参考:
        西门子 vRMS 概念迁移：在振动域中 vRMS 对应振动烈度；
        在电流域中 I_rms 对应负载的等效热电流。

    Args:
        signal: 电流采样序列（应为去 DC 偏置后的信号）

    Returns:
        RMS 值 (归一化后无量纲，原始信号单位为 A)
    """
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_peak_factor(signal: np.ndarray) -> float:
    """
    计算峰值因子 (Crest Factor, CF)

    定义:
        CF = I_peak / I_rms = max(|i|) / I_rms

    电弧检测意义:
        - 纯正弦波: CF ≈ √2 ≈ 1.414
        - 电弧波形: CF 显著升高（因电弧电流存在大量尖峰毛刺）
        - 这是区分电弧和正常负载的**核心特征之一**

    借鉴西门子 DKW 思想:
        CF 本身就是无量纲比值，天然具备跨设备泛化能力。

    Args:
        signal: 电流采样序列

    Returns:
        峰值因子（无量纲）
    """
    rms = compute_rms(signal)
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(signal)) / rms)


def compute_form_factor(signal: np.ndarray) -> float:
    """
    计算波形因子 (Form Factor)

    定义:
        FF = I_rms / I_avg   (其中 I_avg = (1/N) Σ|i_k|)

    物理意义:
        - 纯正弦波: FF = π/(2√2) ≈ 1.111
        - 非线性负载（如整流电路）: FF 偏离 1.111
        - 严重畸变: FF >> 1.111

    Args:
        signal: 电流采样序列

    Returns:
        波形因子（无量纲）
    """
    avg_abs = np.mean(np.abs(signal))
    if avg_abs < 1e-12:
        return 0.0
    return float(compute_rms(signal) / avg_abs)


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    计算峭度 (Kurtosis)

    定义:
        Kurt = E[(i-μ)⁴] / σ⁴

    物理意义:
        - 正态分布: Kurt = 3 (excess kurtosis = 0)
        - 电弧/冲击型异常: Kurt >> 3 (重尾分布，尖峰脉冲多)
        - 平顶波形: Kurt < 3

    在轴承故障诊断中，峭度是检测初期剥落的黄金指标；
    在电流检测中，峭度对电弧脉冲和开关电源高频噪声同样敏感。

    Args:
        signal: 电流采样序列

    Returns:
        峭度 (excess kurtosis，减3后的值)
    """
    n = len(signal)
    if n < 4:
        return 0.0

    mu = np.mean(signal)
    sigma = np.std(signal)

    if sigma < 1e-12:
        return 0.0

    # Fisher 定义（减3，使得正态分布=0）
    kurt = np.mean(((signal - mu) / sigma) ** 4) - 3.0
    return float(kurt)


def compute_skewness(signal: np.ndarray) -> float:
    """
    计算偏度 (Skewness)

    定义:
        Skew = E[(i-μ)³] / σ³

    物理意义:
        - 对称波形: Skew ≈ 0
        - 正向冲击主导（如开关管正向导通过冲）: Skew > 0
        - 负向冲击主导（如半波整流故障）: Skew < 0

    借鉴西门子箱线图偏度算法:
        skewness_approx = (q75 + q25 - 2·q50) / (q75 - q25)

    Args:
        signal: 电流采样序列

    Returns:
        偏度（无量纲）
    """
    n = len(signal)
    if n < 3:
        return 0.0

    mu = np.mean(signal)
    sigma = np.std(signal)

    if sigma < 1e-12:
        return 0.0

    return float(np.mean(((signal - mu) / sigma) ** 3))


def compute_zero_crossing_rate(signal: np.ndarray, tolerance: float = 0.02) -> float:
    """
    计算过零率（归一化的过零次数）

    定义:
        ZCR = (过零点数) / (总采样点数 - 1)

    电弧检测意义（核心特征之一）:
        串联电弧的一个典型时域特征是"平肩部"（Shoulder）——
        在电流过零点附近，电弧电流会在零附近维持一小段时间（平肩），
        然后再快速爬升。这导致:
        - 正常负载: 过零干脆，ZCR 接近理论值
        - 电弧故障: 过零区出现平台，有效过零次数减少

    参考 GB 14287.4-2014 附录A.3.2 电流零休特征。

    Args:
        signal: 电流采样序列
        tolerance: 过零点判定容差（归一化后幅值）

    Returns:
        过零率 [0, 1]
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < 2:
        return 0.0

    # 统计过零次数
    crossings = np.sum(
        (signal[:-1] < -tolerance) & (signal[1:] > tolerance) |
        (signal[:-1] > tolerance) & (signal[1:] < -tolerance)
    )
    return float(crossings / (n - 1))


def compute_differential_stats(signal: np.ndarray) -> Dict[str, float]:
    """
    计算差分统计特征

    对电流波形的一阶差分 ΔI[k] = I[k] - I[k-1] 进行统计分析，
    用于捕捉瞬态突变事件。

    电弧特征:
        电弧电流波形包含大量不规则的突变毛刺，
        ΔI 的方差和最大值会显著大于正常负载。

    返回:
        dict with keys:
          - diff_mean: 差分均值
          - diff_std: 差分标准差
          - diff_max: 差分最大绝对值
          - diff_rms: 差分的 RMS
    """
    if len(signal) < 2:
        return {"diff_mean": 0.0, "diff_std": 0.0, "diff_max": 0.0, "diff_rms": 0.0}

    diff = np.diff(signal)

    return {
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "diff_max": float(np.max(np.abs(diff))),
        "diff_rms": float(np.sqrt(np.mean(np.square(diff)))),
    }


def compute_current_zero_duration(
    signal: np.ndarray,
    sample_rate: float,
    tolerance: float = 0.05,
) -> float:
    """
    计算电流零休时间（电弧"平肩部"持续时间）

    直接检测电流在过零点附近维持在零附近的时间长度。

    电弧诊断:
        - 正常负载: 零休时间 ≈ 0（快速穿越零点）
        - 串联电弧: 零休时间 > 0（电弧在零点熄灭→重燃需要时间）
        - GB 14287.4 将此作为电弧的核心判据之一

    Args:
        signal: 电流采样序列（已归一化）
        sample_rate: 采样率 (Hz)
        tolerance: 判定为"零附近"的幅值阈值

    Returns:
        零休时间 (毫秒)
    """
    signal = np.asarray(signal)
    near_zero = np.abs(signal) < tolerance

    # 找最长连续零区段
    max_duration = 0
    current_duration = 0

    for is_zero in near_zero:
        if is_zero:
            current_duration += 1
        else:
            max_duration = max(max_duration, current_duration)
            current_duration = 0
    max_duration = max(max_duration, current_duration)

    # 转换采样点数 → 毫秒
    return float(max_duration / sample_rate * 1000.0)


# ============================================================
# 批量特征提取
# ============================================================

@dataclass
class TimeDomainFeatures:
    """时域特征集合"""
    rms: float = 0.0
    peak_factor: float = 0.0
    form_factor: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    zero_crossing_rate: float = 0.0
    diff_mean: float = 0.0
    diff_std: float = 0.0
    diff_max: float = 0.0
    diff_rms: float = 0.0
    zero_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rms": self.rms,
            "peak_factor": self.peak_factor,
            "form_factor": self.form_factor,
            "kurtosis": self.kurtosis,
            "skewness": self.skewness,
            "zero_crossing_rate": self.zero_crossing_rate,
            "diff_mean": self.diff_mean,
            "diff_std": self.diff_std,
            "diff_max": self.diff_max,
            "diff_rms": self.diff_rms,
            "zero_duration_ms": self.zero_duration_ms,
        }

    def to_array(self) -> np.ndarray:
        """转为 numpy 数组（用于 ML 模型输入）"""
        return np.array(list(self.to_dict().values()), dtype=np.float32)


def extract_time_domain_features(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    enabled_features: Optional[list] = None,
) -> TimeDomainFeatures:
    """
    从电流信号中提取所有时域特征

    Args:
        signal: 预处理后的电流采样序列
        sample_rate: 采样率 (Hz)
        enabled_features: 启用的特征列表，None 表示全部启用
            e.g. ["rms", "peak_factor", "kurtosis"]

    Returns:
        TimeDomainFeatures 对象
    """
    if enabled_features is None:
        enabled_features = [
            "rms", "peak_factor", "form_factor", "kurtosis",
            "skewness", "zero_crossing_rate", "differential_stats",
            "zero_duration",
        ]

    feats = TimeDomainFeatures()

    if "rms" in enabled_features:
        feats.rms = compute_rms(signal)

    if "peak_factor" in enabled_features:
        feats.peak_factor = compute_peak_factor(signal)

    if "form_factor" in enabled_features:
        feats.form_factor = compute_form_factor(signal)

    if "kurtosis" in enabled_features:
        feats.kurtosis = compute_kurtosis(signal)

    if "skewness" in enabled_features:
        feats.skewness = compute_skewness(signal)

    if "zero_crossing_rate" in enabled_features:
        feats.zero_crossing_rate = compute_zero_crossing_rate(signal)

    if "differential_stats" in enabled_features:
        diff_stats = compute_differential_stats(signal)
        feats.diff_mean = diff_stats["diff_mean"]
        feats.diff_std = diff_stats["diff_std"]
        feats.diff_max = diff_stats["diff_max"]
        feats.diff_rms = diff_stats["diff_rms"]

    if "zero_duration" in enabled_features:
        feats.zero_duration_ms = compute_current_zero_duration(signal, sample_rate)

    return feats
