"""特征提取模块"""
from .time_domain import (
    TimeDomainFeatures,
    extract_time_domain_features,
    compute_rms,
    compute_peak_factor,
    compute_kurtosis,
    compute_zero_crossing_rate,
)
from .frequency_domain import (
    FrequencyDomainFeatures,
    extract_frequency_domain_features,
    compute_spectrum,
    compute_thd,
    compute_high_freq_energy_ratio,
    compute_band_energies,
    compute_sideband_energy,
)
from .time_frequency import (
    TimeFrequencyFeatures,
    extract_time_frequency_features,
    dwt_decompose,
    hilbert_envelope,
)
from .statistical import (
    StatisticalFeatures,
    extract_statistical_features,
    compute_boxplot_summary,
    detect_outliers_iqr,
)
from .three_phase import (
    extract_three_phase_features,
    phase_balance,
    phase_angle_errors,
    symmetry_components,
    phase_correlation,
)
from .cadence import (
    extract_fast_features,
    extract_slow_features,
)

__all__ = [
    # 时域
    "TimeDomainFeatures",
    "extract_time_domain_features",
    "compute_rms",
    "compute_peak_factor",
    "compute_kurtosis",
    "compute_zero_crossing_rate",
    # 频域
    "FrequencyDomainFeatures",
    "extract_frequency_domain_features",
    "compute_spectrum",
    "compute_thd",
    "compute_high_freq_energy_ratio",
    "compute_band_energies",
    "compute_sideband_energy",
    # 时频域
    "TimeFrequencyFeatures",
    "extract_time_frequency_features",
    "dwt_decompose",
    "hilbert_envelope",
    # 统计
    "StatisticalFeatures",
    "extract_statistical_features",
    "compute_boxplot_summary",
    "detect_outliers_iqr",
    # 三相 (v2)
    "extract_three_phase_features",
    "phase_balance",
    "phase_angle_errors",
    "symmetry_components",
    "phase_correlation",
    # 多节拍编排 (v2)
    "extract_fast_features",
    "extract_slow_features",
]
