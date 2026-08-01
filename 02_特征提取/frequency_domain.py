"""
频域特征提取模块

通过 FFT 将时域电流波形转换到频域，提取频谱特征。

核心特征:
  1. 基波幅值/相位 — 工频 50Hz 分量
  2. 各次谐波幅值 — 3/5/7/9/11 次等关键谐波
  3. THD 总谐波失真 — 评价波形正弦度的综合指标
  4. 高频能量比 — 电弧在 >2kHz 频段能量异常升高
  5. 频谱统计量 — 频谱质心、带宽、峰度

借鉴西门子频谱分析的方法论:
  - 结构化频段划分（Bands Allocation）
  - 汉宁窗加窗 FFT
  - 频段能量积分 → 标量风险指数

参考:
  - IEEE 519 谐波标准
  - IEC 61000-4-7 谐波测量
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# FFT 频谱生成
# ============================================================

def compute_spectrum(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    window: str = "hanning",
    output_type: str = "magnitude",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算信号的单边幅度/功率谱

    借鉴西门子:
      - 使用汉宁窗（Hanning）抑制频谱泄露
      - 输出归一化的单边谱

    Args:
        signal: 时域信号
        sample_rate: 采样率 (Hz)
        window: 窗函数类型 ("hanning" / "hamming" / "blackman" / "none")
        output_type: 输出类型 ("magnitude" / "power" / "db")

    Returns:
        (frequencies, spectrum) — 频率轴和对应谱值
    """
    n = len(signal)
    signal = np.asarray(signal, dtype=np.float64)

    # 去均值（避免 DC 泄露到邻近频段）
    signal = signal - np.mean(signal)

    # 加窗
    if window == "hanning":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = np.ones(n)

    # 计算 FFT
    fft_result = np.fft.rfft(signal * w)
    freq = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    if output_type == "magnitude":
        # 幅度谱（归一化）
        spectrum = np.abs(fft_result) * 2.0 / n
        # DC 分量不乘2
        spectrum[0] = np.abs(fft_result[0]) / n
        # Nyquist 分量处理
        if n % 2 == 0:
            spectrum[-1] = np.abs(fft_result[-1]) / n

    elif output_type == "power":
        spectrum = (np.abs(fft_result) * 2.0 / n) ** 2
        spectrum[0] = (np.abs(fft_result[0]) / n) ** 2
        if n % 2 == 0:
            spectrum[-1] = (np.abs(fft_result[-1]) / n) ** 2

    elif output_type == "db":
        mag = np.abs(fft_result) * 2.0 / n
        mag[0] = np.abs(fft_result[0]) / n
        if n % 2 == 0:
            mag[-1] = np.abs(fft_result[-1]) / n
        spectrum = 20 * np.log10(np.maximum(mag, 1e-12))

    else:
        raise ValueError(f"未知输出类型: {output_type}")

    return freq, spectrum


# ============================================================
# 谐波分析
# ============================================================

def find_peak_frequency(
    freq: np.ndarray,
    spectrum: np.ndarray,
    search_range: Tuple[float, float] = (45.0, 55.0),
) -> float:
    """
    在指定频率范围内搜索峰值频率

    用于找到实际电网频率（可能偏离标称 50Hz）。

    Args:
        freq: 频率轴
        spectrum: 幅度谱
        search_range: 搜索范围 (Hz)

    Returns:
        峰值频率 (Hz)
    """
    mask = (freq >= search_range[0]) & (freq <= search_range[1])
    if not np.any(mask):
        return 50.0  # 默认返回标称值
    idx = np.argmax(spectrum[mask])
    return float(freq[mask][idx])


def extract_harmonics(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
    harmonic_orders: List[int] = None,
) -> Dict[int, float]:
    """
    提取各次谐波幅值

    使用加窗插值 FFT 提高精度（类似 IEC 61000-4-7）。

    Args:
        signal: 时域信号
        sample_rate: 采样率 (Hz)
        nominal_freq: 标称基波频率 (Hz)
        harmonic_orders: 需要提取的谐波次数，如 [1, 3, 5, 7, 9, 11]

    Returns:
        {谐波次数: 幅值, ...} 字典
    """
    if harmonic_orders is None:
        harmonic_orders = [1, 3, 5, 7, 9, 11]

    # 先粗估实际基波频率
    freq, mag = compute_spectrum(signal, sample_rate, window="hanning")
    actual_freq = find_peak_frequency(freq, mag)

    results = {}
    freq_resolution = freq[1] - freq[0]  # 频率分辨率

    for order in harmonic_orders:
        target_freq = actual_freq * order
        # 找到最接近的 FFT bin
        idx = int(round(target_freq / freq_resolution))
        if 0 <= idx < len(mag):
            results[order] = float(mag[idx])
        else:
            results[order] = 0.0

    return results


def compute_thd(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
    max_harmonic: int = 21,
) -> float:
    """
    计算总谐波失真 (THD)

    THD = sqrt(Σ(I_h²)) / I_1  （h=2,3,...,max_harmonic）

    IEEE 519 标准:
      - THD < 5%: 正常
      - THD 5%~8%: 轻微畸变
      - THD > 8%: 严重畸变，可能故障

    电弧电流的 THD 通常远高于正常负载。

    Args:
        signal: 时域信号
        sample_rate: 采样率
        nominal_freq: 标称基波
        max_harmonic: 最高谐波次数

    Returns:
        THD 值 (0~1, 乘以100得百分比)
    """
    harmonics = extract_harmonics(
        signal, sample_rate, nominal_freq,
        harmonic_orders=list(range(1, max_harmonic + 1))
    )

    i1 = harmonics.get(1, 0.0)
    if i1 < 1e-12:
        return 0.0

    harmonic_sum_sq = sum(
        harmonics.get(h, 0.0) ** 2
        for h in range(2, max_harmonic + 1)
    )

    return float(np.sqrt(harmonic_sum_sq) / i1)


def compute_high_freq_energy_ratio(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    cutoff_freq: float = 2000.0,
) -> float:
    """
    计算高频能量占比

    这是电弧检测的关键频域特征之一。

    物理依据:
        串联电弧电流包含丰富的高频分量（2kHz~10kHz），
        而正常阻性/感性负载的电流能量集中在低频（工频+低次谐波）。

    Args:
        signal: 时域信号
        sample_rate: 采样率
        cutoff_freq: 高频/低频分界频率 (Hz)

    Returns:
        高频能量占总能量的比例 [0, 1]
    """
    freq, power = compute_spectrum(signal, sample_rate, output_type="power")

    # 低频能量 (DC ~ cutoff)
    low_mask = freq <= cutoff_freq
    low_energy = np.sum(power[low_mask])

    # 高频能量 (cutoff ~ Nyquist)
    high_mask = freq > cutoff_freq
    high_energy = np.sum(power[high_mask])

    total_energy = low_energy + high_energy
    if total_energy < 1e-20:
        return 0.0

    return float(high_energy / total_energy)


# ============================================================
# 频谱统计特征
# ============================================================

def compute_spectral_statistics(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
) -> Dict[str, float]:
    """
    计算频谱统计特征（类似时域的统计量迁移到频域）

    频谱质心:
        类似"频域的均值"，描述能量分布的中心频率。
        电弧信号高频分量增多 → 频谱质心上移。

    频谱带宽:
        能量分布的频带宽度。

    Args:
        signal: 时域信号
        sample_rate: 采样率

    Returns:
        dict with: centroid, bandwidth, skewness, kurtosis, rolloff
    """
    freq, mag = compute_spectrum(signal, sample_rate)

    # 仅使用正幅度进行统计
    if np.sum(mag) < 1e-12:
        return {"centroid": 0.0, "bandwidth": 0.0, "skewness": 0.0,
                "kurtosis": 0.0, "rolloff": 0.0}

    # 频谱质心
    centroid = np.sum(freq * mag) / np.sum(mag)

    # 频谱带宽（以质心为中心的二阶矩）
    bandwidth = np.sqrt(
        np.sum(((freq - centroid) ** 2) * mag) / np.sum(mag)
    )

    # 频谱偏度
    spec_skew = np.sum(((freq - centroid) ** 3) * mag) / (
        np.sum(mag) * (bandwidth ** 3)
    ) if bandwidth > 1e-12 else 0.0

    # 频谱峭度
    spec_kurt = np.sum(((freq - centroid) ** 4) * mag) / (
        np.sum(mag) * (bandwidth ** 4)
    ) if bandwidth > 1e-12 else 0.0

    # 频谱滚降（85% 能量所在的频率）
    cumsum = np.cumsum(mag)
    total = cumsum[-1]
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = float(freq[min(rolloff_idx, len(freq) - 1)])

    return {
        "centroid": float(centroid),
        "bandwidth": float(bandwidth),
        "skewness": float(spec_skew),
        "kurtosis": float(spec_kurt),
        "rolloff": float(rolloff),
    }


# ============================================================
# 频段能量监控（Band Energy）
# ============================================================

def compute_band_energies(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    bands: Optional[List[Tuple[float, float]]] = None,
) -> List[float]:
    """
    结构化频段能量积分

    借鉴西门子 Bands Allocation 监控策略:
      将全频段划分为若干子带，每个子带输出一个标量能量值，
      实现频域的"特征降维"。

    默认频段（针对电流检测）:
      Band 1: 工频附近 [45, 55] Hz    — 基波能量
      Band 2: 低次谐波 [100, 350] Hz  — 3/5/7次谐波
      Band 3: 中高频 [500, 2000] Hz   — 开关噪声
      Band 4: 高频 [2000, 5000] Hz    — 电弧特征频段
      Band 5: 超高频 [5000, 10000] Hz — 超高频EMI

    Args:
        signal: 时域信号
        sample_rate: 采样率
        bands: 自定义频段列表 [(low, high), ...]

    Returns:
        各频段能量列表
    """
    if bands is None:
        bands = [
            (45, 55),      # 基波
            (100, 350),    # 低次谐波
            (500, 2000),   # 中高频
            (2000, 5000),  # 高频
            (5000, 10000), # 超高频
        ]

    freq, power = compute_spectrum(signal, sample_rate, output_type="power")
    energies = []

    for low, high in bands:
        mask = (freq >= low) & (freq <= high)
        band_energy = float(np.sqrt(np.sum(power[mask])))  # RMS 形式
        energies.append(band_energy)

    return energies


# ============================================================
# 批量特征提取
# ============================================================

@dataclass
class FrequencyDomainFeatures:
    """频域特征集合"""
    fundamental_magnitude: float = 0.0
    fundamental_frequency: float = 50.0
    harmonics: Dict[int, float] = field(default_factory=dict)
    thd: float = 0.0
    high_freq_energy_ratio: float = 0.0
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    band_energies: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fundamental_magnitude": self.fundamental_magnitude,
            "fundamental_frequency": self.fundamental_frequency,
            **{f"harmonic_{k}": v for k, v in self.harmonics.items()},
            "thd": self.thd,
            "high_freq_energy_ratio": self.high_freq_energy_ratio,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bandwidth": self.spectral_bandwidth,
            **{f"band_{i}_energy": e for i, e in enumerate(self.band_energies)},
        }

    def to_array(self) -> np.ndarray:
        """转为固定长度的特征向量"""
        harmonic_values = [self.harmonics.get(h, 0.0) for h in [1, 3, 5, 7, 9, 11]]
        band_values = self.band_energies if len(self.band_energies) == 5 else [0.0] * 5

        return np.array([
            self.fundamental_magnitude,
            self.fundamental_frequency,
            *harmonic_values,
            self.thd,
            self.high_freq_energy_ratio,
            self.spectral_centroid,
            self.spectral_bandwidth,
            *band_values,
        ], dtype=np.float32)


def extract_frequency_domain_features(
    signal: np.ndarray,
    sample_rate: float = 50000.0,
    nominal_freq: float = 50.0,
    harmonic_orders: List[int] = None,
) -> FrequencyDomainFeatures:
    """
    从电流信号中提取所有频域特征

    Args:
        signal: 预处理后的电流采样序列
        sample_rate: 采样率 (Hz)
        nominal_freq: 标称工频 (Hz)
        harmonic_orders: 需要提取的谐波次数

    Returns:
        FrequencyDomainFeatures 对象
    """
    if harmonic_orders is None:
        harmonic_orders = [1, 3, 5, 7, 9, 11]

    feats = FrequencyDomainFeatures()

    # 基波频率和幅值
    freq, mag = compute_spectrum(signal, sample_rate)
    feats.fundamental_frequency = find_peak_frequency(freq, mag)
    feats.fundamental_magnitude = float(mag[
        np.argmin(np.abs(freq - feats.fundamental_frequency))
    ])

    # 谐波
    feats.harmonics = extract_harmonics(
        signal, sample_rate, nominal_freq, harmonic_orders
    )

    # THD
    feats.thd = compute_thd(signal, sample_rate, nominal_freq)

    # 高频能量比
    feats.high_freq_energy_ratio = compute_high_freq_energy_ratio(
        signal, sample_rate, cutoff_freq=2000.0
    )

    # 频谱统计
    spec_stats = compute_spectral_statistics(signal, sample_rate)
    feats.spectral_centroid = spec_stats["centroid"]
    feats.spectral_bandwidth = spec_stats["bandwidth"]

    # 频段能量
    feats.band_energies = compute_band_energies(signal, sample_rate)

    return feats


# ============================================================
# MCSA 转子条边带特征 (v2 新增)
# ============================================================

def compute_sideband_energy(
    signal: np.ndarray,
    sample_rate: float = 16000.0,
    f1: float = 50.0,
    slip: float = 0.03,
    width: float = 0.5,
) -> Dict[str, float]:
    """
    MCSA 转子条边带能量特征（慢路径用）

    转子条故障（broken rotor bar）在电流谱基频两侧产生边带：
      f_side = f1 ± 2·k·s·f1  (k=1,2,...；s=转差率)

    ⚠️ 物理约束：边带与基频可能仅差 0.5~2Hz，需要长窗（≥2~4s）
       使频率分辨率 Δf = fs/N 足够小——因此本函数只应在慢路径调用。

    Args:
        signal: 时域信号（长窗）
        sample_rate: 采样率 (Hz)
        f1: 基波频率（变频器输出频率）
        slip: 假定转差率
        width: 边带搜索半宽 (Hz)

    Returns:
        {sideband_upper, sideband_lower, sideband_total, sideband_ratio}
    """
    freq, power = compute_spectrum(signal, sample_rate, output_type="power")

    def _band_energy(center: float) -> float:
        mask = (freq >= center - width) & (freq <= center + width)
        return float(np.sum(power[mask]))

    e_base = _band_energy(f1)
    e_upper = _band_energy(f1 + 2.0 * slip * f1)
    e_lower = _band_energy(f1 - 2.0 * slip * f1)
    e_total = e_upper + e_lower

    return {
        "sideband_upper": e_upper,
        "sideband_lower": e_lower,
        "sideband_total": e_total,
        "sideband_ratio": float(e_total / e_base) if e_base > 1e-20 else 0.0,
    }
