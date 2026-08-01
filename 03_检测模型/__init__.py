"""检测模型模块"""
# ── v1 方向（电弧/NILM，保留参考，非 v2 目标） ──
from .arc_detection import (
    ArcDetectionCNN,
    ArcDetectionCNN_LSTM,
    create_arc_model,
    arc_detection_rule_based,
)
from .load_identification import (
    LoadClassifier1DResNet,
    create_load_classifier,
    LOAD_CLASSES,
    LOAD_CLASSES_EN,
)

# ── v2 保留（P3 正常建模 + 电能质量） ──
from .anomaly_detection import (
    CurrentAutoEncoder,
    AnomalyDetector,
    create_anomaly_detector,
)
from .power_quality import (
    analyze_power_quality,
    PowerQualityReport,
    PowerQualityStatus,
)

# ── v2 新增（过程状态识别，复刻打底） ──
from .process_state import (
    ProcessState,
    StateRuleConfig,
    ProcessStateClassifier,
    classify_state,
)

__all__ = [
    # 电弧检测（v1 参考）
    "ArcDetectionCNN",
    "ArcDetectionCNN_LSTM",
    "create_arc_model",
    "arc_detection_rule_based",
    # 负载识别（v1 参考）
    "LoadClassifier1DResNet",
    "create_load_classifier",
    "LOAD_CLASSES",
    "LOAD_CLASSES_EN",
    # 异常检测（P3）
    "CurrentAutoEncoder",
    "AnomalyDetector",
    "create_anomaly_detector",
    # 电能质量
    "analyze_power_quality",
    "PowerQualityReport",
    "PowerQualityStatus",
    # 过程状态（v2）
    "ProcessState",
    "StateRuleConfig",
    "ProcessStateClassifier",
    "classify_state",
]
