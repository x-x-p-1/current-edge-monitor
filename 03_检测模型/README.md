# 03 — 检测模型模块手册

## 模块概述

检测模型是算法栈的**第三层**，将特征向量/原始波形送入 AI 模型或规则引擎，输出检测结论。

```
波形/特征 → 过程状态(规则) + 异常检测(AE) + 电能质量 → 检测结果
          （v2 目标）      （P3）            （保留）
```

## v2 变更记录（对齐三相变频电机方向）

| 变更 | v1 | v2 |
|------|----|----|
| **目标方向** | 电弧检测(AFCI) / 负载识别(NILM)（市电单相） | **过程状态识别**（三相变频电机，过程管理） |
| **新增 `process_state.py`** | 无 | 规则基线过程状态分类器：STOP/TRANSIENT/IDLE/LOAD/STALL，自适应基线（DKW 思路），快路径无标签可跑 |
| **异常检测 AE** | 保留 | **保留（即 P3 正常建模雏形）** |
| **电能质量** | 保留 | **保留**（THD/谐波/频率对 VFD 电机同样适用） |
| 电弧 / 负载识别 | 主目标 | **归为 v1 方向参考**，文件保留不动（含其测试） |

> v2 的"规则基线 + AI 级联"架构思想不变：`process_state`（规则，快路径）作初筛与状态上下文，
> AE（无监督）作异常兜底，后续 ML 模型（监督分类）接替或叠加。
> 基线学习终极形态由数学家 P3（无监督正常建模）接替。

**核心设计哲学**：
- **多模型并行**：主模型+辅助模型+规则基线，互为补充
- **NPU 优先**：模型设计时就考虑 INT8 量化和 RKNN 兼容性
- **规则兜底**：当 AI 模型无法加载时（如 NPU 驱动故障），规则基线保证基本检测能力

---

## 模型矩阵

| 模型 | 架构 | 参数量 | NPU延迟 | 训练需求 | 优先级 |
|------|------|--------|---------|---------|--------|
| **ArcCNN** | 1D-CNN (5层) | ~50K | <0.3ms | 标注数据集 | ⭐⭐⭐ |
| ArcCNN+LSTM | CNN + BiLSTM | ~300K | ~1ms | 标注数据集 | ⭐⭐ |
| **AutoEncoder** | Conv AE | ~80K | <0.4ms | 仅需正常数据 | ⭐⭐⭐ |
| **LoadClassifier** | 1D-ResNet | ~200K | ~0.6ms | 标注数据集 | ⭐⭐ |
| **Rule Baseline** | 规则引擎 | N/A | <0.1ms | 无需训练 | ⭐⭐⭐ |

---

## 模型一：串联电弧检测 CNN

### 网络结构

```
Input [1, 1, 256]  — 原始电流波形
    ↓
Conv1D(1→16, k5) → BN → ReLU → MaxPool(2)    [1, 16, 128]
    ↓
Conv1D(16→32, k5) → BN → ReLU → MaxPool(2)   [1, 32, 64]
    ↓
Conv1D(32→64, k3) → BN → ReLU → MaxPool(2)   [1, 64, 32]
    ↓
Conv1D(64→128, k3) → BN → ReLU               [1, 128, 32]
    ↓
AdaptiveAvgPool1d → [1, 128]
    ↓
FC(128→64) → ReLU → Dropout(0.3)
    ↓
FC(64→2) → Softmax → [正常概率, 电弧概率]
```

### 设计要点

| 设计决策 | 理由 |
|----------|------|
| 3 层 MaxPool(2) | 逐步压缩时间维度，8×下采样 |
| AdaptiveAvgPool | 替换 Flatten → 减少参数量，支持可变输入长度 |
| BN + Dropout | 防止小数据集过拟合 |
| 仅 50K 参数 | INT8 量化后 < 25KB，适合边缘部署 |

### 使用示例

```python
from 03_检测模型.arc_detection import create_arc_model

model = create_arc_model("cnn", input_length=256)
model.load_state_dict(torch.load("models/arc_cnn.pth"))

# 推理
x = torch.randn(1, 1, 256)  # [batch, channels, length]
probs, preds = model.predict(x)  # probs: [0.92, 0.08], preds: 0(=正常)
```

---

## 模型二：规则基线检测

### 规则集（GB 14287.4 启发）

| 规则 | 阈值 | 物理依据 |
|------|------|----------|
| 峰值因子 > 3.0 | 正常正弦波 CF≈1.4 | 电弧尖峰 → CF 飙升 |
| 高频能量比 > 0.08 | 正常负载 HF≈0 | 电弧 → 2kHz+ 能量增大 |
| 零休时间 > 0.5ms | 正常过零瞬间通过 | 电弧平肩 → 零休延长 |
| 离群比例 > 15% | 正常信号离群≈5% | 电弧毛刺 → 大量离群点 |

### 仲裁逻辑

```
满足 ≥ 2 条规则 → 判定为电弧
Confidence = 满足规则数 / 4
```

### 与 AI 模型的关系

```
[规则基线] —— 快速初筛（<0.1ms）
     ↓ 可能异常
[ArcCNN] —— 精确判定（<0.3ms）
     ↓ 确认
[后处理决策] —— 防抖确认 + 动作
```

```python
from 03_检测模型.arc_detection import arc_detection_rule_based

is_arc, confidence, rule_details = arc_detection_rule_based(clean_signal, 50000)
# rule_details = {
#     "peak_factor":    {"value": 4.2, "triggered": True},
#     "high_freq_ratio": {"value": 0.12, "triggered": True},
#     ...
# }
```

---

## 模型三：无监督异常检测 AutoEncoder

### 为什么需要 AE

- 不需要标注故障数据（标注成本极高）
- 对**未知类型异常**也有检测能力（如新型电力电子故障）
- 作为 CNN 的补充——“即使 CNN 认为正常，AE 发现异常也要注意”

### 网络结构

```
编码器: 1×256 → 16×128 → 32×64 → 64×32 → 128×16 → FC→32(潜在空间)
解码器: 32 → FC→128×16 → 64×32 → 32×64 → 16×128 → 1×256
```

### 异常判定逻辑

1. **训练**：仅用正常负载波形训练 AE
2. **校准**：计算正常样本的重构误差分布 $(\mu_{err}, \sigma_{err})$
3. **推理**：重构误差 $z = \frac{\text{MSE} - \mu_{err}}{\sigma_{err}}$
4. **判定**：$z > 3.0$ → 异常

$$\text{Anomaly} \iff \text{MSE}_{recon} > \mu_{normal} + 3\sigma_{normal}$$

### 借鉴西门子 DKW 思想

AE 的重构误差基线校准机制与西门子 DKW 的自适应基线学习同源：
- DKW = $a_{RMS} / a_{RMS,ref}$ —— 相对健康基线的倍率偏离
- AE Score = $(\text{MSE} - \mu) / \sigma$ —— 相对正常分布的 Z-Score 偏离

```python
from 03_检测模型.anomaly_detection import create_anomaly_detector

detector = create_anomaly_detector(input_length=256)
detector.calibrate(normal_samples)  # 基线学习
scores, is_anomaly = detector.predict(x)  # scores: z-score, is_anomaly: bool
```

---

## 模型四：负载类型识别 1D-ResNet

### 网络结构

```
Conv1D(1→32, k7, s2) → MaxPool → ResBlock×2(32→64) → ResBlock×2(64→128)
→ ResBlock×2(128→256) → GAP → FC(256→8)
```

### 负载类别

| ID | 中文 | English | 典型设备 |
|----|------|---------|---------|
| 0 | 阻性负载 | resistive | 电热器、白炽灯 |
| 1 | 感性负载 | inductive | 电机、变压器 |
| 2 | 容性负载 | capacitive | 电容补偿柜 |
| 3 | 开关电源 | switching PSU | LED、充电器 |
| 4 | 变频器 | VFD drive | 变频空调 |
| 5 | 整流电路 | rectifier | 电镀电源 |
| 6 | 混合负载 | mixed | 多设备组合 |
| 7 | 未知 | unknown | 无法识别 |

```python
from 03_检测模型.load_identification import create_load_classifier, LOAD_CLASSES

model = create_load_classifier(num_classes=8)
probs, preds = model.predict(x)  # preds: [2] → LOAD_CLASSES[2] = "容性负载"
```

---

## 模型五：电能质量分析

遵循 IEC 61000-4-30 A 级方法，计算以下指标：

| 指标 | 算法 | 标准 |
|------|------|------|
| 频率偏差 | 插值 FFT + 抛物线拟合 | IEC 61000-4-30 |
| 谐波 THD | 加窗 FFT + 谐波提取 | IEEE 519 |
| 间谐波 | 谐波间频段峰值搜索 | IEC 61000-4-7 |
| 闪变 Pst | 简化 IEC 61000-4-15 | 半周期 RMS + 带通 + 统计 |
| 三相不平衡 | 对称分量法 | GB/T 15543 |

### THD 三级状态

| 状态 | THD 范围 | 动作 |
|------|---------|------|
| NORMAL | < 8% | 无 |
| WARNING | 8–15% | 记录日志 |
| ALARM | > 15% | 报警 |

```python
from 03_检测模型.power_quality import analyze_power_quality

report = analyze_power_quality(signal, sample_rate=50000)
print(report.to_dict())
# {"frequency": 50.02, "thd": 12.3, "thd_status": "warning", ...}
```

---

## 模型训练建议

| 步骤 | 数据量 | 说明 |
|------|--------|------|
| 规则基线验证 | 100+ 帧 | 先跑通规则，确认特征有效 |
| CNN 训练 | 5000+ 帧（含标注） | 需要正常+电弧标注数据 |
| AE 训练 | 10000+ 帧（仅正常） | 无需标注，数据获取容易 |
| 负载识别 | 每种 500+ 帧 | 需要多种负载的标注数据 |

---

## 参考标准

- **GB 14287.4-2014** — 故障电弧探测器标准
- **UL 1699** — Arc-Fault Circuit-Interrupters
- **IEC 61000-4-30** — 电能质量测量
- **IEEE 519** — 谐波控制
- **西门子 CMS** — 三级预警 + 自适应基线
