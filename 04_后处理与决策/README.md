# 04 — 后处理与决策融合模块手册

## 模块概述

后处理与决策是算法栈的**第四层**，将检测模型的原始输出转化为可靠、稳定的决策结论。

```
模型输出 → 分数平滑 → 迟滞防抖 → 多模型融合 → 事件聚合 → 动作建议
```

## v2 说明（对齐三相变频电机方向）

本层**直接复用，无需重构**——它实现的正是在本项目架构中承担
**"触发确认 + 分级决策 + 事件聚合"** 的角色：

| 本层组件 | 在本项目中的角色 |
|------|------|
| `HysteresisAlarm`（迟滞+连续确认） | 触发引擎的确认机制（防误报/报警风暴） |
| `MultiLevelHysteresisAlarm` | 三级状态（NORMAL/WARNING/ALARM）→ 过程管理分级 |
| `ScoreSmoother` | 快路径分数平滑 |
| `DecisionFusionEngine` | 多模型/多特征融合（规则+AE+ML 级联） |
| `EventAggregator` | 事件切片聚合（对齐"触发切片捕获"） |

> 输入由 v2 的 `03.process_state` 状态分数 / `03.anomaly_detection` 异常分数 /
> 慢路径诊断分数提供，输出接入事件切片与追溯档案。

**核心设计哲学**（借鉴西门子 APL MonAnL 防抖机制）：

> 宁可延迟 10ms 报警，也绝不因单帧噪声触发一次误停机。

---

## 文件结构

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `hysteresis.py` | 迟滞报警 | 状态机 + 连续确认 + 迟滞量 |
| `decision_fusion.py` | 决策融合 | 加权投票、级联决策、分数平滑 |

---

## 核心算法一：迟滞报警 (HysteresisAlarm)

### 问题

高频检测（每 2.56ms 一帧）下，模型输出可能因噪声在阈值附近抖动：

```
时间 →
帧1: 0.88 → 超阈值
帧2: 0.79 → 低于阈值（噪声抖动）
帧3: 0.91 → 超阈值
帧4: 0.82 → 低于阈值
```

如果一超过阈值就报警 → **报警风暴（Alarm Storm）**

### 解决方案：连续确认 + 迟滞

借鉴西门子 APL `MonAnL` 功能块的状态机设计：

```
                    score > upper
    ┌─────────┐  连续 confirm_count 帧  ┌─────────┐
    │ NORMAL  │ ───────────────────────→ │  ALARM  │
    └─────────┘                          └─────────┘
         ↑                                    │
         │   score < lower                    │
         └── 连续 release_count 帧 ───────────┘
```

#### 数学表达

$$\text{State}_k = \begin{cases}
\text{ALARM},  & \sum_{i=0}^{N_c-1} \mathbb{I}(\text{score}_{k-i} > L_{upper}) = N_c \\[6pt]
\text{NORMAL}, & \sum_{i=0}^{N_r-1} \mathbb{I}(\text{score}_{k-i} < L_{lower}) = N_r \\[6pt]
\text{State}_{k-1}, & \text{otherwise}
\end{cases}$$

其中 $L_{lower} = L_{upper} \times 0.85$（迟滞带）。

### 参数选择

| 参数 | 推荐值 | 效果 |
|------|--------|------|
| `confirm_count` | 3 | 3帧×2.56ms ≈ 7.7ms 确认延迟 |
| `release_count` | 5 | 5帧≈12.8ms 后才解除报警 |
| `threshold_lower` | upper×0.85 | 15% 迟滞带 |

### 使用示例

```python
from 04_后处理与决策.hysteresis import HysteresisAlarm, AlarmState

alarm = HysteresisAlarm(
    threshold_upper=0.85,
    threshold_lower=0.70,
    confirm_count=3,
    release_count=5,
)

for score in detection_stream:
    state = alarm.update(score)
    if state == AlarmState.ALARM:
        trigger_protection()
    elif state == AlarmState.NORMAL:
        resume_normal()
```

---

## 核心算法二：多级迟滞报警 (MultiLevelHysteresisAlarm)

借鉴西门子 CMS2000 的三级状态：

| 级别 | DKW 范围（西门子原版） | 电流域映射 |
|------|----------------------|-----------|
| **NORMAL** | DKW ≤ 2.0 | score ≤ 0.5 |
| **WARNING** | 2.0 < DKW ≤ 4.0 | 0.5 < score ≤ 0.85 |
| **ALARM** | DKW > 4.0 | score > 0.85 |

```python
alarm = MultiLevelHysteresisAlarm(
    warning_threshold=0.5,
    alarm_threshold=0.85,
    confirm_count=3,
    hysteresis_ratio=0.10,  # 10% 迟滞
)
```

---

## 核心算法三：事件聚合 (EventAggregator)

### 问题

一次持续 500ms 的电弧故障会产生 500/2.56 ≈ **195 帧报警**。如果不做聚合，日志和控制信号会被洪水淹没。

### 解决方案

```
时间轴 →  [事件1: 帧10~帧50]  ...正常...  [事件2: 帧120~帧200]

合并窗口 = 500ms: 间隔 < 500ms 的报警帧合并为同一事件
```

```python
from 04_后处理与决策.hysteresis import EventAggregator

aggregator = EventAggregator(merge_window_ms=500.0, frame_interval_ms=2.56)

for frame_idx, is_alarm in enumerate(alarm_stream):
    aggregator.update(is_alarm, metadata={"rms": current_rms})

events = aggregator.finalize()
# events = [
#     {"start_frame": 30, "end_frame": 80, "duration_frames": 51, ...},
# ]
```

---

## 核心算法四：分数平滑 (ScoreSmoother)

消除模型输出的逐帧高频抖动：

| 方法 | 公式 | 延迟 | 适用 |
|------|------|------|------|
| 移动平均 | $\bar{s}_k = \frac{1}{W}\sum_{i=k-W+1}^{k} s_i$ | W/2 帧 | 通用 |
| 指数平滑 | $\bar{s}_k = \alpha s_k + (1-\alpha)\bar{s}_{k-1}$ | ~1/α 帧 | 快速响应 |

```python
from 04_后处理与决策.decision_fusion import ScoreSmoother

smoother = ScoreSmoother(window_size=5, method="moving_average")
for raw_score in model_outputs:
    stable_score = smoother.update(raw_score)
```

---

## 核心算法五：多模型决策融合 (DecisionFusionEngine)

### 四种融合策略

| 策略 | 逻辑 | 适用场景 |
|------|------|----------|
| **投票** | 多数决定 | 模型数量多且独立 |
| **加权投票** | $\sum w_i \cdot s_i$ | ⭐ 不同模型可靠性不同 |
| **最大置信度** | 取最自信的模型 | 单一主导模型 |
| **级联** | 快筛 → 精判 | ⭐ 规则+AI 组合 |

### 级联策略（推荐）

```
[规则基线]  ──正常──→ 输出: NORMAL
     │
     可能异常
     ▼
[ArcCNN]  ──正常──→ 输出: MONITOR
     │
     确认异常
     ▼
输出: EMERGENCY_SHUTDOWN
```

### 推荐动作映射

| 置信度 | 动作 | 含义 |
|--------|------|------|
| > 0.9 | `emergency_shutdown` | 极高置信度 → 立即断路 |
| 0.7–0.9 | `alarm_and_log` | 高置信度 → 声光报警+记录 |
| 0.5–0.7 | `log_and_monitor` | 中等置信度 → 记录日志持续监控 |
| < 0.5 | `none` | 正常 |

```python
from 04_后处理与决策.decision_fusion import (
    DecisionFusionEngine, FusionMethod, ModelPrediction
)

engine = DecisionFusionEngine(
    method=FusionMethod.WEIGHTED,
    weights={"arc_cnn": 0.5, "anomaly_ae": 0.3, "rule_based": 0.2},
)

preds = [
    ModelPrediction("arc_cnn", is_anomaly=True, confidence=0.92, raw_score=0.92),
    ModelPrediction("anomaly_ae", is_anomaly=True, confidence=0.78, raw_score=4.2),
    ModelPrediction("rule_based", is_anomaly=True, confidence=0.65, raw_score=0.65),
]

result = engine.fuse(preds)
print(result.to_dict())
# {"is_anomaly": true, "overall_confidence": 0.85, "recommended_action": "alarm_and_log"}
```

---

## 完整管线示例

```python
from 04_后处理与决策.hysteresis import HysteresisAlarm, EventAggregator, AlarmState
from 04_后处理与决策.decision_fusion import DecisionFusionEngine, ScoreSmoother, FusionMethod, ModelPrediction

smoother = ScoreSmoother(window_size=5)
alarm = HysteresisAlarm(threshold_upper=0.85, confirm_count=3, release_count=5)
aggregator = EventAggregator(merge_window_ms=500.0)
fusion = DecisionFusionEngine(method=FusionMethod.WEIGHTED,
                               weights={"arc_cnn": 0.5, "anomaly_ae": 0.3, "rule_based": 0.2})

for frame_idx, raw_waveform in enumerate(adc_stream):
    clean = preprocessor.process(raw_waveform)
    
    # 并行推理
    arc_score = arc_model.predict(clean)
    ae_score = ae_detector.predict(clean)
    rule_conf, _ = rule_based_check(clean)
    
    # 分数平滑
    arc_smooth = smoother.update(arc_score)
    
    # 多模型融合
    preds = [
        ModelPrediction("arc_cnn", arc_score > 0.5, arc_score, arc_score),
        ModelPrediction("anomaly_ae", ae_score > 3.0, ae_score, ae_score),
        ModelPrediction("rule_based", rule_conf > 0.5, rule_conf, rule_conf),
    ]
    result = fusion.fuse(preds)
    
    # 迟滞确认
    state = alarm.update(result.overall_confidence)
    aggregator.update(state == AlarmState.ALARM)

# 最终获取所有事件
events = aggregator.finalize()
```

---

## 参考来源

- **西门子 APL MonAnL** — 迟滞防抖状态机
- **西门子 CMS2000** — 三级状态 (OK/Warning/Alarm)
- **IEC 61131-3** — PLC 报警管理范式
