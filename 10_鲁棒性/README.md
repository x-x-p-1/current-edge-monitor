# 10 — 鲁棒性（运行时加固）

## 模块概述

对齐《08_鲁棒性/鲁棒性清单.md》的软件实现，为采集/预处理/检测链路提供
**数值守卫 + 输入质量自评 + 混沌注入验证**，让设备"能犯病、能识别、能恢复"。

设计原则：**算法必须在被监测设备坏之前不能坏。** 任何一层失败都不允许静默。

## 文件结构

| 文件 | 功能 | 对齐目标 |
|------|------|----------|
| `guards.py` | 数值守卫：NaN/Inf 清洗、除零/对数守卫、`NumericGuard` | R1 / R2 |
| `input_quality.py` | 输入质量自评：削波 / 直流偏置 / 断线 / 丢样 / 缺相 / 相位错序 | R3 / R4 / R5 / R6 |
| `chaos.py` | 混沌注入器（§4.2 fault injection） | §4.2 |
| `engine.py`（09） | AcquisitionEngine 内嵌：NaN 防护 + 质量门控 + 单帧隔离 | R1 / R7 / R10 |

## 目标矩阵对照（《08_鲁棒性/鲁棒性清单.md》）

| # | 威胁 | 本模块实现 | 验收 |
|---|------|-----------|------|
| R1 | NaN/Inf 传播 | `sanitize_nan_inf` / `NumericGuard`；engine 入环前清洗 | 注入后不崩溃、标记、恢复 |
| R2 | 除零 / 病态计算 | `safe_divide` / `safe_log10` | 无 NaN 进入状态机 |
| R3 | ADC 削波 | `InputQuality.clipped`（削波率 vs 契约 full_scale） | 标记"失真帧"而非误诊谐波 |
| R4 | 断线 / 直流偏置 | `dropout`（极低方差）/ `dc_bias`（均值偏移） | 断线 → 输入不可用，不误报堵转 |
| R5 | 丢样 / 时戳不连续 | `sample_drop`（时戳连续性） | 丢样 → 标记数据段 |
| R6 | 缺相 / 相位错序 | `phase_lost`（RMS 不平衡）/ `phase_reversed`（FFT 相位序） | 识别为缺相而非负载异常 |
| R7 | 内存无界增长 | engine 质量日志 `deque(maxlen)` 有界 | 无线性增长 |
| R10 | 未捕获异常 | engine 单帧隔离（`try/except` + 计数） | 注入任意异常，进程存活 100% |

## 关键设计

### 1. 质量自评在原始域（输入层）
断线/偏置/削波在**原始采样**上检测（块级），与滤波状态无关。
若对预处理后帧检测，从正常切到常值断线时，慢基线状态瞬态会让输出非零，
掩盖断线（var 不再为 0）。分层防护的第一层必须作用在输入。

### 2. 硬无效 vs 软劣化
- **硬无效**（NaN/Inf、断线）：`valid=False` → engine 保存原始但**跳过诊断**（降级，不误报）
- **软劣化**（削波/偏置/丢样/缺相/错序）：`valid=True` 但 `score` 降低 →
  engine 记日志"标记 + 继续"（鲁棒性清单：输入劣化标记 + 继续）

### 3. 单帧隔离（R10）
`AcquisitionEngine.feed` 内每帧 `try/except`：任何单帧异常（含非数组输入、
标量、空块）→ 记录 `_errors` 与质量日志，进程存活，后续帧继续工作。

### 4. 混沌注入（§4.2）
`ChaosInjector` 支持：`nan / inf / dropout / dc_bias / clip / sample_drop /
phase_swap / exception`。验证每类注入"不崩溃 + 被标记 + 可恢复"。

## 快速开始

```bash
# 混沌注入全种类验证（8 类：存活/标记/恢复 报告）
python 07_测试工具/verify_robustness.py

# 回归测试（10 鲁棒性 23 用例）
python -m pytest 06_测试与验证/test_robustness.py -q
```

## 集成方式

```python
from 10_鲁棒性 import InputQuality, ChaosInjector
from 09_采集层 import AcquisitionEngine, WatchdogFeatures, TriggerEngine, SliceCapture, CaptureContext

engine = AcquisitionEngine(
    pp, wd, trig, cap,
    input_quality=InputQuality(channels=3, full_scale=3.3),  # 契约满量程
)
# feed 自动：NaN 清洗 → 质量门控 → 看门狗/触发；异常单帧隔离
```

## 测试状态

`06_测试与验证/test_robustness.py`：23 用例（数值守卫 4 / 输入质量 8 / 混沌 7 /
引擎鲁棒性 4）。全量回归见仓库 README。
