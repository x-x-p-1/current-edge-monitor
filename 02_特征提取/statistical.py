"""
统计特征提取模块

借鉴西门子 LGF 库的统计学算法族:
  - 箱线图 (Boxplot) 五数概括
  - 四分位距 (IQR)
  - 基于 IQR 的离群点检测
  - 箱线图偏度近似
  - 直方图频数统计
  - 最小二乘线性回归（趋势分析）

这些统计方法在电流信号分析中的价值:
  1. 电流波形往往不满足正态分布假设 → 需要非参数统计
  2. 箱线图方法对野值鲁棒
  3. 四分位数比均值/方差更稳健地反映信号分布特性
  4. 回归趋势斜率可用于劣化评估
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 箱线图统计 (Boxplot)
# ============================================================

def compute_quartiles(
    data: np.ndarray,
) -> Tuple[float, float, float]:
    """
    计算下四分位数 q25、中位数 q50、上四分位数 q75

    借鉴西门子 LGF_Boxplot 的算法:
        使用排序后直接索引法（method='inclusive'），
        适合嵌入式实时核。

    Args:
        data: 一维信号序列

    Returns:
        (q25, q50, q75)
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    if n < 4:
        return float(sorted_data[0]), float(np.median(sorted_data)), float(sorted_data[-1])

    # 使用线性插值法（numpy 默认 method='linear'，与西门子近似）
    q25 = float(np.percentile(sorted_data, 25))
    q50 = float(np.median(sorted_data))
    q75 = float(np.percentile(sorted_data, 75))

    return q25, q50, q75


def compute_boxplot_summary(data: np.ndarray) -> Dict[str, float]:
    """
    计算箱线图五数概括 + IQR + 离群边界

    借鉴西门子 LGF_Boxplot 的输出结构:
      min, q25, q50, q75, max, IQR, lower_bound, upper_bound

    Args:
        data: 一维信号序列

    Returns:
        箱线图统计字典
    """
    q25, q50, q75 = compute_quartiles(data)
    iqr = q75 - q25

    # Tukey 离群边界 (1.5 IQR)
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    return {
        "min": float(np.min(data)),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "max": float(np.max(data)),
        "iqr": float(iqr),
        "lower_fence": float(lower_bound),
        "upper_fence": float(upper_bound),
    }


def compute_boxplot_skewness(data: np.ndarray) -> float:
    """
    基于四分位数的偏度近似计算

    西门子公式:
        skewness = (q75 + q25 - 2*q50) / (q75 - q25)

    物理意义:
        - ≈ 0: 分布对称
        - > 0: 正偏（右尾长）
        - < 0: 负偏（左尾长）

    相比传统的三阶矩偏度，四分位偏度对野值鲁棒得多，
    在电流信号这种经常含突变脉冲的场景中更可靠。

    Args:
        data: 一维信号序列

    Returns:
        四分位偏度 [-1, 1]
    """
    q25, q50, q75 = compute_quartiles(data)
    denom = q75 - q25
    if abs(denom) < 1e-12:
        return 0.0
    return float((q75 + q25 - 2 * q50) / denom)


# ============================================================
# 离群点检测
# ============================================================

def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = 1.5,
) -> Tuple[np.ndarray, int]:
    """
    基于 IQR 的离群点检测

    借鉴西门子 LGF_Boxplot 内置的 Outlier Detection 功能:
      - rangeOutlier=1.5: 温和离群值
      - rangeOutlier=3.0: 极端离群值

    电弧检测中的应用:
        电弧电流波形中含有大量"离群"采样点（尖峰毛刺），
        统计离群点数占比可以作为电弧强度的一个量化指标。

    Args:
        data: 一维信号序列
        multiplier: IQR 乘数 (1.5 = 温和, 3.0 = 极端)

    Returns:
        (outlier_mask, outlier_count)
        outlier_mask: bool 数组，标记离群点位置
    """
    q25, q50, q75 = compute_quartiles(data)
    iqr = q75 - q25

    lower = q25 - multiplier * iqr
    upper = q75 + multiplier * iqr

    mask = (data < lower) | (data > upper)
    return mask, int(np.sum(mask))


def outlier_ratio(data: np.ndarray, multiplier: float = 1.5) -> float:
    """
    计算离群点比例

    Args:
        data: 一维信号序列
        multiplier: IQR 乘数

    Returns:
        离群点比例 [0, 1]
    """
    _, count = detect_outliers_iqr(data, multiplier)
    return float(count / len(data)) if len(data) > 0 else 0.0


# ============================================================
# 直方图统计
# ============================================================

def compute_histogram_stats(
    data: np.ndarray,
    num_bins: int = 20,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    计算直方图频数分布

    借鉴西门子 LGF_Histogram 的统计方法:
      - 将采样点按幅值区间统计频数
      - 输出可用于概率密度估计

    电流信号中的应用:
        - 正常正弦电流的幅值直方图呈U型（两端频率高）
        - 电弧电流的直方图更分散，甚至多峰

    Args:
        data: 一维信号序列
        num_bins: 区间数量
        range_min: 下限
        range_max: 上限

    Returns:
        {"bin_edges": array, "counts": array, "pdf": array}
    """
    if range_min is None:
        range_min = float(np.min(data))
    if range_max is None:
        range_max = float(np.max(data))

    counts, bin_edges = np.histogram(data, bins=num_bins, range=(range_min, range_max))
    pdf = counts / (len(data) + 1e-12)

    return {
        "bin_edges": bin_edges,
        "counts": counts,
        "pdf": pdf,
    }


# ============================================================
# 线性回归趋势分析
# ============================================================

def linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """
    最小二乘线性回归 y = m*x + c

    借鉴西门子 LGF_RegressionLine 的最小二乘公式。

    在电流检测中的应用:
        对连续多帧的特征值（如 RMS、THD）做趋势分析:
        - m > 0 且持续增大: 劣化加速
        - m ≈ 0: 稳定运行

    Args:
        x: 自变量序列（如时间索引）
        y: 因变量序列（如每帧的 RMS 值）

    Returns:
        (slope_m, intercept_c, r_squared)
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if len(y) > 0 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 斜率 m
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-12:
        return 0.0, y_mean, 0.0

    m = numerator / denominator
    c = y_mean - m * x_mean

    # 决定系数 R²
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return float(m), float(c), float(r2)


def compute_trend_slope(
    feature_history: np.ndarray,
    window: int = 10,
) -> float:
    """
    计算特征的时间趋势斜率

    用于实时劣化评估——监控特征值（如 RMS、THD、离群比例）
    是否随时间单调上升。

    Args:
        feature_history: 最近 N 帧的特征值数组
        window: 回归窗口大小

    Returns:
        趋势斜率 m
    """
    if len(feature_history) < 2:
        return 0.0

    # 取最近的 window 个值
    recent = feature_history[-min(window, len(feature_history)):]
    x = np.arange(len(recent), dtype=np.float64)
    m, _, _ = linear_regression(x, recent)
    return m


# ============================================================
# 统计特征汇总
# ============================================================

@dataclass
class StatisticalFeatures:
    """统计特征集合"""
    # 箱线图
    boxplot: Dict[str, float] = field(default_factory=dict)

    # 偏度
    quartile_skewness: float = 0.0

    # 离群点
    outlier_ratio_15iqr: float = 0.0
    outlier_ratio_30iqr: float = 0.0

    # 直方图特征
    histogram_entropy: float = 0.0

    # 趋势
    trend_slope: float = 0.0

    def to_dict(self) -> dict:
        result = {}
        result.update(self.boxplot)
        result["quartile_skewness"] = self.quartile_skewness
        result["outlier_ratio_15iqr"] = self.outlier_ratio_15iqr
        result["outlier_ratio_30iqr"] = self.outlier_ratio_30iqr
        result["histogram_entropy"] = self.histogram_entropy
        result["trend_slope"] = self.trend_slope
        return result

    def to_array(self) -> np.ndarray:
        """转为固定长度特征向量"""
        bp = self.boxplot
        feat_list = [
            bp.get("min", 0.0), bp.get("q25", 0.0), bp.get("q50", 0.0),
            bp.get("q75", 0.0), bp.get("max", 0.0),
            bp.get("iqr", 0.0), bp.get("lower_fence", 0.0), bp.get("upper_fence", 0.0),
            self.quartile_skewness,
            self.outlier_ratio_15iqr,
            self.outlier_ratio_30iqr,
            self.histogram_entropy,
            self.trend_slope,
        ]
        return np.array(feat_list, dtype=np.float32)


def extract_statistical_features(
    signal: np.ndarray,
    feature_history: Optional[np.ndarray] = None,
) -> StatisticalFeatures:
    """
    从电流信号中提取统计特征

    Args:
        signal: 预处理后的电流采样序列
        feature_history: 历史特征值数组（用于趋势计算），可选

    Returns:
        StatisticalFeatures 对象
    """
    feats = StatisticalFeatures()

    # 箱线图
    feats.boxplot = compute_boxplot_summary(signal)

    # 四分位偏度
    feats.quartile_skewness = compute_boxplot_skewness(signal)

    # 离群点比例
    feats.outlier_ratio_15iqr = outlier_ratio(signal, multiplier=1.5)
    feats.outlier_ratio_30iqr = outlier_ratio(signal, multiplier=3.0)

    # 直方图熵
    hist = compute_histogram_stats(signal)
    pdf = hist["pdf"]
    eps = 1e-12
    feats.histogram_entropy = float(-np.sum(pdf * np.log(pdf + eps)))

    # 趋势斜率
    if feature_history is not None and len(feature_history) > 1:
        feats.trend_slope = compute_trend_slope(feature_history)

    return feats
