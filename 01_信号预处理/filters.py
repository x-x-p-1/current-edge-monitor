"""
数字滤波器模块 (v2 — 三相/VFD 方向重构)
========================================
实现电流检测所需的滤波器，支持单相 (N,) 与三相 (N, C) 数据：
  - 直流偏置去除（滑动均值减法 / 高通 RC 耦合模拟）
  - 巴特沃斯带通滤波（核心处理，截止频率可配置）
  - 移动平均平滑
  - 陷波滤波（去特定频率，如工频）
  - Savitzky-Golay 多项式平滑

v2 设计变更（对齐三相变频电机方向）:
  1. 通道结构：统一约定 (N, C) = 时间×通道，所有滤波沿时间轴逐通道处理；
     单通道 (N,) 完全兼容。
  2. 带通参数：v1 硬编码 45Hz~2500Hz（市电/电弧方向）会①杀掉变频电机低速
     基频（f1 可低至几 Hz）②滤掉 >2.5kHz 的诊断频带，与高频特征自相矛盾。
     v2 默认 lowcut=5Hz / highcut=4000Hz，均可配置；设计约束为 highcut
     不得低于诊断频带上限，PWM 纹波由模拟抗混叠处理。

参考西门子 SM1281 的 AFE 设计哲学：
  高通阻断直流漂移 → 低通滤除高频噪声 → 保留特征频段
"""

import numpy as np
from scipy import signal
from typing import Tuple


def _per_channel_axis0(func, data, *args, **kwargs):
    """沿时间轴(axis=0)逐通道应用 func；单通道 (N,) 直接处理。

    数据约定：(N,) 单通道 或 (N, C) 时间×通道。
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return func(data, *args, **kwargs)
    if data.ndim == 2:
        return np.column_stack(
            [func(data[:, ch], *args, **kwargs) for ch in range(data.shape[1])]
        )
    raise ValueError(f"不支持的数据维度: {data.ndim}")


def remove_dc_offset(data: np.ndarray, window: int = 100) -> np.ndarray:
    """
    滑动均值法去除直流偏置

    原理（类比西门子 AFE 的高通 RC 耦合）:
      计算局部滑动均值作为 DC 估计值，从原始信号中减去，
      等效于一阶高通滤波。

    数学表达:
      I_ac[k] = I_raw[k] - (1/W) * Σ(i=k-W+1..k) I_raw[i]

    Args:
        data: 原始采样数据，shape (N,) 或 (N, C)（时间×通道）
        window: 滑动均值窗口大小

    Returns:
        去除 DC 偏置后的信号，shape 同输入
    """
    return _per_channel_axis0(_remove_dc_1d, data, window)


def _remove_dc_1d(data: np.ndarray, window: int) -> np.ndarray:
    """单通道滑动均值去直流（内部实现）"""
    if len(data) < window:
        # 数据不足窗口大小时，使用全局均值
        dc_estimate = np.mean(data)
        return data - dc_estimate

    # 卷积实现滑动平均
    kernel = np.ones(window) / window
    dc_estimate = np.convolve(data, kernel, mode="same")

    # 边界处理：前 window/2 和后 window/2 点使用局部均值
    half_w = window // 2
    dc_estimate[:half_w] = np.mean(data[:window])
    dc_estimate[-half_w:] = np.mean(data[-window:])

    return data - dc_estimate


def butter_bandpass_coefficients(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    设计巴特沃斯带通滤波器系数

    设计约束（三相变频电机方向）:
      - lowcut 必须低于 VFD 最低输出频率（低速时基频可低至几 Hz），
        否则会杀掉低速基频 → 默认 5.0Hz
      - highcut 不得低于诊断频带上限（MCSA 边带/谐波/包络频带），
        否则会滤掉诊断信息（v1 的 2500Hz 即犯此错误）→ 默认 4000Hz
      - PWM 开关纹波主要由模拟抗混叠处理，不在数字带通里一刀切

    Args:
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数（越高滚降越陡峭）

    Returns:
        (b, a) 滤波器系数，用于 scipy.signal.filtfilt / lfilter
    """
    nyquist = 0.5 * fs
    if lowcut >= highcut:
        raise ValueError(f"lowcut({lowcut}) 必须小于 highcut({highcut})")

    # 归一化截止频率
    low = lowcut / nyquist
    high = highcut / nyquist

    # 带通滤波器设计 (4阶 = 2阶带通 × 2)
    b, a = signal.butter(order, [low, high], btype="band")

    return b, a


def bandpass_filter(
    data: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """
    应用零相位带通滤波

    使用 filtfilt 实现零相位偏移（forward-backward filtering），
    这在电流波形分析中非常重要——不能引入任何相位延迟，
    否则过零点位置偏移会导致电弧"平肩部"特征被扭曲。

    Args:
        data: 输入信号，shape (N,) 或 (N, C)
        b, a: 滤波器系数

    Returns:
        滤波后信号（零相位），shape 同输入
    """
    return _per_channel_axis0(_bandpass_1d, data, b, a)


def _bandpass_1d(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """单通道零相位带通滤波（内部实现）"""
    return signal.filtfilt(b, a, data)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    简单移动平均平滑

    递推高效实现（借鉴西门子 LGF_SimpleSmoothingFB）:
      y[k] = y[k-1] + (x[k] - x[k-N]) / N

    Args:
        data: 输入信号，shape (N,) 或 (N, C)
        window: 平滑窗口大小

    Returns:
        平滑后信号，shape 同输入
    """
    return _per_channel_axis0(_moving_average_1d, data, window)


def _moving_average_1d(data: np.ndarray, window: int) -> np.ndarray:
    """单通道滑动平均（内部实现）"""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def notch_filter(
    data: np.ndarray,
    freq: float,
    fs: float,
    q: float = 30.0,
) -> np.ndarray:
    """
    陷波滤波器 — 去除特定频率（如 50Hz 工频）

    在某些应用中可能需要去除基波分量，只分析谐波和瞬态特征。

    Args:
        data: 输入信号，shape (N,) 或 (N, C)
        freq: 目标陷波频率 (Hz)
        fs: 采样率 (Hz)
        q: 品质因数（越大越窄）

    Returns:
        滤波后信号，shape 同输入
    """
    b, a = signal.iirnotch(freq / (0.5 * fs), q)
    return _per_channel_axis0(_notch_1d, data, b, a)


def _notch_1d(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """单通道陷波滤波（内部实现）"""
    return signal.filtfilt(b, a, data)


def savitzky_golay_smooth(
    data: np.ndarray,
    window_length: int = 5,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Savitzky-Golay 多项式平滑滤波

    借鉴西门子 LGF_SmoothByPolynom 的设计:
      对滑动窗口进行最小二乘多项式拟合，在滤除高频噪声的同时
      极度保留波形的峰值和幅值特征——非常适合在电弧检测等场景下
      保留突变的峰值信息。

    5点3次多项式卷积核（与西门子一致）:
      y[n] = (-3*x[n-2] + 12*x[n-1] + 17*x[n] + 12*x[n+1] - 3*x[n+2]) / 35

    Args:
        data: 输入信号，shape (N,) 或 (N, C)
        window_length: 窗口长度（必须为奇数），默认5
        polyorder: 多项式阶数，默认3

    Returns:
        平滑后信号，shape 同输入
    """
    if window_length % 2 == 0:
        window_length += 1  # 必须为奇数
    return _per_channel_axis0(_savgol_1d, data, window_length, polyorder)


def _savgol_1d(data: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """单通道 Savitzky-Golay 平滑（内部实现）"""
    return signal.savgol_filter(data, window_length, polyorder)
