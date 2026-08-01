"""
电能质量分析模块

基于 IEC 61000-4-30 标准的电能质量指标计算。

分析项目:
  1. 频率偏差
  2. 电压/电流 RMS
  3. 谐波分析 (THD, 各次谐波)
  4. 间谐波检测
  5. 闪变 (Flicker) — IEC 61000-4-15
  6. 三相不平衡度

在边缘检测场景中，电能质量分析作为周期性监控任务，
为电弧检测和负载识别提供辅助判断依据。

参考:
  - IEEE 519 谐波控制
  - IEC 61000-4-30 A 级测量方法
  - GB/T 14549 电能质量 公用电网谐波
"""

import importlib

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# 数字开头目录（01_/02_）无法用标准相对/绝对导入，运行时用 importlib 加载
_align_mod = importlib.import_module("01_信号预处理.alignment")
_freq_mod = importlib.import_module("02_特征提取.frequency_domain")
_time_mod = importlib.import_module("02_特征提取.time_domain")


# ============================================================
# 频率估计
# ============================================================

def estimate_frequency_zero_crossing(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
) -> float:
    """
    基于过零点检测的频率估计

    统计 N 个完整周期的总采样点数，反算实际频率:
        f = N_cycles * sample_rate / total_samples

    参考 IEC 61000-4-30: 10 个周期的频率测量（50Hz 系统）

    Args:
        signal: 时域信号
        sample_rate: 采样率
        nominal_freq: 标称频率

    Returns:
        估计的电网频率 (Hz)
    """
    find_zero_crossings = _align_mod.find_zero_crossings

    signal = np.asarray(signal, dtype=np.float64)

    # 寻找正向过零点
    crossings = find_zero_crossings(signal, direction="positive", tolerance=0.01)

    if len(crossings) < 3:
        return nominal_freq

    # 使用中间的过零点（避免边界效应）
    if len(crossings) > 10:
        crossings = crossings[2:-2]

    # 计算平均周期
    periods = np.diff(crossings)
    mean_period = np.mean(periods)

    if mean_period < 1:
        return nominal_freq

    return float(sample_rate / mean_period)


def estimate_frequency_interpolated_fft(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
) -> float:
    """
    基于插值 FFT 的高精度频率估计

    使用抛物线插值法对 FFT 峰值附近的三个点进行拟合，
    得到亚 bin 分辨率的频率估计。

    Args:
        signal: 时域信号
        sample_rate: 采样率

    Returns:
        估计频率 (Hz)
    """
    compute_spectrum = _freq_mod.compute_spectrum

    n = len(signal)
    freq, mag = compute_spectrum(signal, sample_rate, window="hanning")

    # 在 45~55 Hz 范围内找峰值
    mask = (freq >= 45) & (freq <= 55)
    if not np.any(mask):
        return 50.0

    idx = np.argmax(mag[mask])
    peak_idx = np.where(mask)[0][idx]

    if peak_idx <= 0 or peak_idx >= len(mag) - 1:
        return float(freq[peak_idx])

    # 抛物线插值: 用 peak-1, peak, peak+1 三点拟合
    alpha = mag[peak_idx - 1]
    beta = mag[peak_idx]
    gamma = mag[peak_idx + 1]

    # 插值偏移量
    delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma) if (alpha - 2 * beta + gamma) != 0 else 0.0

    # 频率分辨率 × 偏移后的索引
    freq_res = freq[1] - freq[0]
    estimated_freq = freq[peak_idx] + delta * freq_res

    return float(estimated_freq)


# ============================================================
# RMS 计算 (符合 IEC 标准)
# ============================================================

def compute_rms_half_cycle(
    signal: np.ndarray,
    sample_rate: float,
    nominal_freq: float = 50.0,
) -> np.ndarray:
    """
    半周期滑动 RMS（IEC 61000-4-30）

    每半个工频周期更新一次 RMS 值，用于检测电压暂降/暂升。

    Args:
        signal: 时域信号
        sample_rate: 采样率
        nominal_freq: 标称频率

    Returns:
        半周期 RMS 序列
    """
    half_cycle_samples = int(sample_rate / (2 * nominal_freq))
    n = len(signal)

    rms_values = []
    for start in range(0, n - half_cycle_samples + 1, half_cycle_samples):
        window = signal[start:start + half_cycle_samples]
        rms = np.sqrt(np.mean(np.square(window)))
        rms_values.append(rms)

    return np.array(rms_values)


# ============================================================
# 间谐波检测
# ============================================================

def detect_interharmonics(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
    threshold_ratio: float = 0.05,
) -> Dict[int, float]:
    """
    检测间谐波（非整数次谐波分量）

    间谐波的存在通常指示:
      - 变频器/电力电子装置的开关频率
      - 电弧的非周期分量
      - 电网谐振

    方法:
        在谐波之间的频段搜索显著峰值。

    Args:
        signal: 时域信号
        sample_rate: 采样率
        nominal_freq: 标称频率
        threshold_ratio: 判定为间谐波的幅值阈值（相对于基波）

    Returns:
        {频率(Hz): 幅值, ...}
    """
    compute_spectrum = _freq_mod.compute_spectrum

    freq, mag = compute_spectrum(signal, sample_rate, window="hanning")

    # 基波幅值
    fundamental_mask = (freq >= nominal_freq - 5) & (freq <= nominal_freq + 5)
    if not np.any(fundamental_mask):
        return {}
    fundamental_mag = np.max(mag[fundamental_mask])

    threshold = fundamental_mag * threshold_ratio

    interharmonics = {}

    # 每次谐波之间搜索间谐波
    for harmonic_order in range(1, 11):
        low = nominal_freq * harmonic_order + 5  # 谐波右侧
        high = nominal_freq * (harmonic_order + 1) - 5  # 下一谐波左侧

        if high <= low:
            continue

        mask = (freq >= low) & (freq <= high)
        if not np.any(mask):
            continue

        # 寻找局部峰值
        band_mag = mag[mask]
        band_freq = freq[mask]

        peaks = []
        for i in range(1, len(band_mag) - 1):
            if band_mag[i] > band_mag[i - 1] and band_mag[i] > band_mag[i + 1]:
                if band_mag[i] > threshold:
                    peaks.append((float(band_freq[i]), float(band_mag[i])))

        for f, m in peaks:
            interharmonics[f] = m

    return interharmonics


# ============================================================
# 闪变 (Flicker) — IEC 61000-4-15 简化版
# ============================================================

def compute_short_term_flicker(
    rms_sequence: np.ndarray,
    sample_rate_equivalent: float,
) -> float:
    """
    短时闪变严重度 Pst 的简化计算

    基于 IEC 61000-4-15 闪变仪方框图，实现简化版:
      - 输入: 半周期 RMS 序列
      - 带通滤波 (0.05~35Hz) → 加权 → 统计分级

    注意: 完整实现需要灯-眼-脑响应模型，此处提供工程近似。

    Args:
        rms_sequence: 半周期 RMS 值序列
        sample_rate_equivalent: RMS 序列的等效采样率 (100Hz for 50Hz system)

    Returns:
        短时闪变 Pst (简化)
    """
    from scipy import signal

    if len(rms_sequence) < 10:
        return 0.0

    # 去均值
    rms_norm = rms_sequence / np.mean(rms_sequence) - 1.0

    # 带通滤波: 0.05 ~ 35 Hz (人类对闪变敏感的频率范围)
    nyquist = sample_rate_equivalent / 2
    low = 0.05 / nyquist
    high = min(35.0 / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype="band")
    flicker_signal = signal.filtfilt(b, a, rms_norm)

    # 加权 (简化: 8.8Hz 中心频率的 A 曲线)
    # 使用 RMS 统计代替完整的分级统计
    pst = 4.0 * np.std(flicker_signal)

    return float(pst)


# ============================================================
# 三相不平衡度
# ============================================================

def compute_unbalance_factor(
    magnitudes: np.ndarray,
    angles_deg: np.ndarray,
) -> float:
    """
    计算三相电压/电流不平衡度（对称分量法）

    正序 I₁ = (I_a + α·I_b + α²·I_c) / 3
    负序 I₂ = (I_a + α²·I_b + α·I_c) / 3
    不平衡度 = |I₂| / |I₁| × 100%

    其中 α = e^(j·120°) = 1∠120°

    Args:
        magnitudes: 三相幅值数组 [phase_a, phase_b, phase_c]
        angles_deg: 三相相位角（度）

    Returns:
        不平衡度百分比
    """
    if len(magnitudes) != 3 or len(angles_deg) != 3:
        return 0.0

    # 转为复数相量
    angles_rad = np.deg2rad(angles_deg)
    phasors = magnitudes * np.exp(1j * angles_rad)

    # 旋转因子
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)

    # 对称分量
    I_positive = (phasors[0] + alpha * phasors[1] + alpha ** 2 * phasors[2]) / 3.0
    I_negative = (phasors[0] + alpha ** 2 * phasors[1] + alpha * phasors[2]) / 3.0

    if abs(I_positive) < 1e-12:
        return 0.0

    return float(abs(I_negative) / abs(I_positive) * 100.0)


# ============================================================
# 电能质量汇总
# ============================================================

class PowerQualityStatus(str, Enum):
    """电能质量状态"""
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"


@dataclass
class PowerQualityReport:
    """电能质量分析报告"""
    # 频率
    frequency: float = 50.0
    frequency_deviation_hz: float = 0.0

    # RMS
    current_rms: float = 0.0
    rms_trend: float = 0.0  # RMS 变化趋势

    # 谐波
    thd: float = 0.0
    thd_status: PowerQualityStatus = PowerQualityStatus.NORMAL
    harmonics: Dict[int, float] = field(default_factory=dict)

    # 间谐波
    interharmonics: Dict[int, float] = field(default_factory=dict)

    # 闪变
    flicker_pst: float = 0.0

    # 不平衡
    unbalance_pct: float = 0.0

    # 综合状态
    overall_status: PowerQualityStatus = PowerQualityStatus.NORMAL

    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency,
            "frequency_deviation_hz": self.frequency_deviation_hz,
            "current_rms": self.current_rms,
            "rms_trend": self.rms_trend,
            "thd": self.thd,
            "thd_status": self.thd_status.value,
            **{f"harmonic_{k}": v for k, v in self.harmonics.items()},
            "interharmonic_count": len(self.interharmonics),
            "flicker_pst": self.flicker_pst,
            "unbalance_pct": self.unbalance_pct,
            "overall_status": self.overall_status.value,
        }


def analyze_power_quality(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
    thd_warning: float = 8.0,
    thd_alarm: float = 15.0,
) -> PowerQualityReport:
    """
    综合电能质量分析

    Args:
        signal: 预处理后的电流波形
        sample_rate: 采样率
        nominal_freq: 标称频率
        thd_warning: THD 预警阈值 (%)
        thd_alarm: THD 报警阈值 (%)

    Returns:
        PowerQualityReport
    """
    compute_rms = _time_mod.compute_rms
    compute_thd = _freq_mod.compute_thd
    extract_harmonics = _freq_mod.extract_harmonics
    compute_spectrum = _freq_mod.compute_spectrum

    report = PowerQualityReport()

    # 频率估计
    report.frequency = estimate_frequency_interpolated_fft(signal, sample_rate)
    report.frequency_deviation_hz = abs(report.frequency - nominal_freq)

    # 电流 RMS
    report.current_rms = float(compute_rms(signal))

    # THD 和谐波
    report.thd = float(compute_thd(signal, sample_rate, nominal_freq)) * 100.0
    report.harmonics = {
        k: v for k, v in extract_harmonics(
            signal, sample_rate, nominal_freq,
            [1, 3, 5, 7, 9, 11, 13, 15]
        ).items()
    }

    # THD 状态判定
    if report.thd > thd_alarm:
        report.thd_status = PowerQualityStatus.ALARM
    elif report.thd > thd_warning:
        report.thd_status = PowerQualityStatus.WARNING
    else:
        report.thd_status = PowerQualityStatus.NORMAL

    # 间谐波
    report.interharmonics = {
        int(k): v for k, v in detect_interharmonics(
            signal, sample_rate, nominal_freq
        ).items()
    }

    # 闪变 (简化为 RMS 序列分析)
    rms_seq = compute_rms_half_cycle(signal, sample_rate, nominal_freq)
    report.flicker_pst = float(compute_short_term_flicker(rms_seq, 100.0))

    # 综合状态
    if report.thd_status == PowerQualityStatus.ALARM:
        report.overall_status = PowerQualityStatus.ALARM
    elif report.thd_status == PowerQualityStatus.WARNING:
        report.overall_status = PowerQualityStatus.WARNING
    else:
        report.overall_status = PowerQualityStatus.NORMAL

    return report
