"""
多模型决策融合模块

将多个检测模型的输出进行融合，产生最终的判定结论。

融合策略（借鉴西门子 CMS 多指标综合判定的思想）:
  1. 投票法 (Voting): 每个模型 1 票，多数决定
  2. 加权投票 (Weighted Voting): 不同模型不同权重
  3. 最大置信度 (Max Confidence): 取置信度最高的模型的结论
  4. 级联 (Cascade): 先快速筛选，再精细判断

在电流检测场景中的应用:
  - 电弧 CNN: 主要判定器，权重最高
  - 异常检测 AE: 辅助判定，捕捉未知异常
  - 电能质量: 提供上下文（如 THD 超高时提升电弧阈值）
  - 规则基线: 作为 AI 模型失效时的 fallback
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 融合方法
# ============================================================

class FusionMethod(str, Enum):
    VOTING = "voting"              # 简单多数投票
    WEIGHTED = "weighted"          # 加权投票
    MAX_CONFIDENCE = "max_confidence"  # 最大置信度
    CASCADE = "cascade"            # 级联


@dataclass
class ModelPrediction:
    """单个模型的预测结果"""
    model_name: str
    is_anomaly: bool               # 是否异常
    confidence: float              # 置信度 [0, 1]
    raw_score: float               # 原始分数
    details: dict = field(default_factory=dict)  # 额外细节

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "raw_score": self.raw_score,
            **self.details,
        }


@dataclass
class FusionResult:
    """融合后的最终结果"""
    is_anomaly: bool
    overall_confidence: float      # 综合置信度 [0, 1]
    method: FusionMethod
    individual_predictions: List[ModelPrediction]
    anomaly_type: str = "unknown"  # 异常类型
    recommended_action: str = "none"  # 推荐动作

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "overall_confidence": self.overall_confidence,
            "method": self.method.value,
            "anomaly_type": self.anomaly_type,
            "recommended_action": self.recommended_action,
            "individual": [p.to_dict() for p in self.individual_predictions],
        }


# ============================================================
# 融合引擎
# ============================================================

class DecisionFusionEngine:
    """
    多模型决策融合引擎

    用法:
        engine = DecisionFusionEngine(
            method=FusionMethod.WEIGHTED,
            weights={"arc_cnn": 0.5, "anomaly_ae": 0.3, "rule_based": 0.2},
        )

        preds = [
            ModelPrediction("arc_cnn", True, 0.92, 0.92),
            ModelPrediction("anomaly_ae", True, 0.78, 4.5),
            ModelPrediction("rule_based", False, 0.40, 0.40),
        ]

        result = engine.fuse(preds)
    """

    def __init__(
        self,
        method: FusionMethod = FusionMethod.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
    ):
        self.method = method
        self.weights = weights or {}
        self.threshold = threshold

    def fuse(
        self,
        predictions: List[ModelPrediction],
    ) -> FusionResult:
        """
        融合多个模型的预测结果

        Args:
            predictions: 各模型的预测结果列表

        Returns:
            FusionResult 融合结果
        """
        if not predictions:
            return FusionResult(
                is_anomaly=False,
                overall_confidence=0.0,
                method=self.method,
                individual_predictions=[],
            )

        if self.method == FusionMethod.VOTING:
            return self._fuse_voting(predictions)
        elif self.method == FusionMethod.WEIGHTED:
            return self._fuse_weighted(predictions)
        elif self.method == FusionMethod.MAX_CONFIDENCE:
            return self._fuse_max_confidence(predictions)
        elif self.method == FusionMethod.CASCADE:
            return self._fuse_cascade(predictions)
        else:
            raise ValueError(f"未知融合方法: {self.method}")

    def _fuse_voting(self, predictions: List[ModelPrediction]) -> FusionResult:
        """简单多数投票"""
        anomaly_votes = sum(1 for p in predictions if p.is_anomaly)
        total = len(predictions)
        is_anomaly = anomaly_votes > total / 2
        confidence = anomaly_votes / total

        return FusionResult(
            is_anomaly=is_anomaly,
            overall_confidence=confidence,
            method=FusionMethod.VOTING,
            individual_predictions=predictions,
            anomaly_type=self._determine_anomaly_type(predictions),
            recommended_action=self._recommend_action(is_anomaly, confidence),
        )

    def _fuse_weighted(self, predictions: List[ModelPrediction]) -> FusionResult:
        """加权投票"""
        total_weight = 0.0
        weighted_score = 0.0

        for p in predictions:
            w = self.weights.get(p.model_name, 1.0 / len(predictions))
            total_weight += w

            # 将置信度映射到 [-1, 1]: 异常为正，正常为负
            signed_confidence = p.confidence if p.is_anomaly else -p.confidence
            weighted_score += w * signed_confidence

        if total_weight > 0:
            weighted_score /= total_weight

        is_anomaly = weighted_score > self.threshold
        # 归一化置信度到 [0, 1]
        overall_confidence = (weighted_score + 1.0) / 2.0
        overall_confidence = max(0.0, min(1.0, overall_confidence))

        return FusionResult(
            is_anomaly=is_anomaly,
            overall_confidence=overall_confidence,
            method=FusionMethod.WEIGHTED,
            individual_predictions=predictions,
            anomaly_type=self._determine_anomaly_type(predictions),
            recommended_action=self._recommend_action(is_anomaly, overall_confidence),
        )

    def _fuse_max_confidence(self, predictions: List[ModelPrediction]) -> FusionResult:
        """最大置信度"""
        best = max(predictions, key=lambda p: p.confidence)

        return FusionResult(
            is_anomaly=best.is_anomaly,
            overall_confidence=best.confidence,
            method=FusionMethod.MAX_CONFIDENCE,
            individual_predictions=predictions,
            anomaly_type=best.model_name if best.is_anomaly else "normal",
            recommended_action=self._recommend_action(best.is_anomaly, best.confidence),
        )

    def _fuse_cascade(self, predictions: List[ModelPrediction]) -> FusionResult:
        """
        级联融合:
          首先用规则基线（快速）筛选
          → 规则判断可能异常时，启动 CNN
          → CNN 高置信度时，确认报警
        """
        # 找规则基线结果
        rule_pred = next((p for p in predictions if p.model_name == "rule_based"), None)

        # 找 CNN 结果
        cnn_pred = next((p for p in predictions if p.model_name == "arc_cnn"), None)

        if rule_pred and rule_pred.is_anomaly:
            if cnn_pred and cnn_pred.is_anomaly:
                # 规则 + CNN 都判断异常 → 高置信度报警
                confidence = max(rule_pred.confidence, cnn_pred.confidence)
                return FusionResult(
                    is_anomaly=True,
                    overall_confidence=confidence,
                    method=FusionMethod.CASCADE,
                    individual_predictions=predictions,
                    anomaly_type="arc_fault_confirmed",
                    recommended_action="emergency_shutdown",
                )
            elif cnn_pred:
                # 规则认为异常但 CNN 不认 → 降低置信度
                return FusionResult(
                    is_anomaly=True,
                    overall_confidence=rule_pred.confidence * 0.6,
                    method=FusionMethod.CASCADE,
                    individual_predictions=predictions,
                    anomaly_type="possible_anomaly",
                    recommended_action="log_and_monitor",
                )
            else:
                # 仅有规则判断异常
                return FusionResult(
                    is_anomaly=True,
                    overall_confidence=rule_pred.confidence * 0.4,
                    method=FusionMethod.CASCADE,
                    individual_predictions=predictions,
                    anomaly_type="rule_triggered",
                    recommended_action="log_and_monitor",
                )
        else:
            # 规则判断正常
            return FusionResult(
                is_anomaly=False,
                overall_confidence=0.1,
                method=FusionMethod.CASCADE,
                individual_predictions=predictions,
                anomaly_type="normal",
                recommended_action="none",
            )

    def _determine_anomaly_type(self, predictions: List[ModelPrediction]) -> str:
        """根据各模型结果推断异常类型"""
        for p in predictions:
            if p.is_anomaly and p.model_name == "arc_cnn" and p.confidence > 0.8:
                return "arc_fault"
            if p.is_anomaly and p.model_name == "anomaly_ae" and p.confidence > 0.8:
                return "unknown_anomaly"
            if p.is_anomaly and p.model_name == "power_quality":
                return "power_quality_issue"

        # 检查是否有任何异常
        if any(p.is_anomaly for p in predictions):
            return "possible_anomaly"
        return "normal"

    @staticmethod
    def _recommend_action(is_anomaly: bool, confidence: float) -> str:
        """根据结果推荐动作"""
        if not is_anomaly:
            return "none"
        if confidence > 0.9:
            return "emergency_shutdown"  # 紧急停机
        elif confidence > 0.7:
            return "alarm_and_log"       # 报警+记录
        elif confidence > 0.5:
            return "log_and_monitor"     # 仅记录
        else:
            return "none"


# ============================================================
# 滑动平均平滑
# ============================================================

class ScoreSmoother:
    """
    检测分数滑动平均平滑器

    对模型输出的逐帧分数做滑动平均，去除单帧抖动。
    """

    def __init__(self, window_size: int = 5, method: str = "moving_average"):
        """
        Args:
            window_size: 平滑窗口大小（帧数）
            method: "moving_average" | "exponential"
        """
        self.window_size = window_size
        self.method = method
        self._buffer = []
        self._ema = None  # 指数移动平均的状态
        self._alpha = 2.0 / (window_size + 1)  # EMA 衰减因子

    def update(self, score: float) -> float:
        """
        输入当前帧分数，返回平滑后的分数

        Args:
            score: 当前帧的原始分数

        Returns:
            平滑后的分数
        """
        self._buffer.append(score)

        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if self.method == "moving_average":
            return float(np.mean(self._buffer))
        elif self.method == "exponential":
            if self._ema is None:
                self._ema = score
            else:
                self._ema = self._alpha * score + (1 - self._alpha) * self._ema
            return float(self._ema)
        else:
            return score

    def reset(self):
        self._buffer = []
        self._ema = None
