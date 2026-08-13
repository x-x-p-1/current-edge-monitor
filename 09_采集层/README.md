# 09 — 采集层（M1 主线，P0）

## 模块概述

面向边缘端实时运行的**采集 / 触发 / 切片捕获**层（TODO B · M1 软件主线）。
在硬件（ADC + RK3588S）就绪前，本层以纯软件 + 仿真数据跑通「全速率环形缓冲 →
毫秒级看门狗特征 → 触发引擎（stopping time）→ 预/后触发切片落盘」闭环，
并可直接挂接 01 预处理流式管线。

**核心动机**（对齐 README 设计理念 §3）：
设备大部分时间正常，异常是稀有事件。架构 = 全速率环形缓冲常开 + 毫秒级看门狗
特征 + 统计触发 → 触发时冻结**预触发**波形、补录**后触发**，整段切片落盘并携带
上下文。**触发事件切片 = 未来 ML 训练数据**（数据飞轮）。

## 文件结构

| 文件 | 功能 |
|------|------|
| `ring_buffer.py` | 全速率环形缓冲：固定容量、常开、O(1) 写入、任意偏移读取（预触发切片） |
| `watchdog.py` | 毫秒级看门狗特征：RMS / 峰值包络 / 峰值因子 / RMS 斜率（快路径） |
| `trigger.py` | 触发引擎：EWMA 基线 + K·σ 统计判据 + 平滑控误报 + 去抖确认 + 迟滞复位 |
| `slice_capture.py` | 切片捕获：预触发 + 后触发，`.npz` 落盘，携带上下文（数据飞轮） |
| `engine.py` | `AcquisitionEngine`：feed 驱动的 M1 运行时主循环（串起全链路） |

## 设计要点

### 1. 环形缓冲（`RingBuffer`）
- 不变式：**绝对索引 i 的样本存于 `_buf[i % capacity]`**（任何写入量级均成立）。
- `write` O(1) 摊销；`slice_range(start, stop)` 按绝对索引取（越界裁剪）；
  `get_last(n)` 取最近 n 点。
- 预触发不完整检测：`first_available_index()` 与 `pre_avail` 对比。

### 2. 看门狗特征（`WatchdogFeatures`）
- 输入：预处理后的**幅值保持**帧（`norm_enabled=False`，见 v2.1 说明）。
- 输出 `WatchdogSnapshot`：三相 `rms` / `envelope` / `crest_factor` + 平均 RMS 的
  `rms_slope`（最近 N 帧最小二乘趋势，快速劣化检测）。
- 性能实测 ~0.09ms/帧（纯 numpy，256×3 @16k），远低于 1.8ms 目标。

### 3. 触发引擎（`TriggerEngine`）
- **自适应基线**：EMA 估计 μ 与波动 σ，仅稳态帧更新（异常帧不污染基线 → 对齐
  08 鲁棒性"状态机防漂移"）。
- **触发判据**：对 EWMA 平滑后的判据量 `z=(x-μ)/σ` 超过 ±K 且连续 `confirm_count`
  帧 → 触发（平滑控单帧毛刺误报；K=4 → 正常误触率 ≈ 6e-5/帧）。
- **复位判据**：对**原始值**连续 `release_count` 帧回到基线内 → 复位（快速迟滞，
  不被平滑拖慢）。
- 数学 ARL 保证属于 TODO E · P1 触发统计，本实现为工程基线。

### 4. 切片捕获（`SliceCapture`）
- 触发点 = 环形缓冲当前写入点；`[trigger - pre, trigger + post)` 冻结为切片。
- 落盘 `.npz`（float32 + 上下文）：`sample_rate / channels / pre/post / reason /
  timestamp / meta`。
- 输出目录默认 `slices/`（`CaptureContext.out_dir` 可配）。

### 5. 采集引擎（`AcquisitionEngine`）
```python
engine = AcquisitionEngine(pp, wd, trig, cap)   # pp: CurrentPreprocessor 等
for chunk in adc_stream:
    saved = engine.feed(chunk, t_s=t, meta={"state": "LOAD", "vfd_freq": 50.0})
    for info in saved:
        print(info["path"], info["pre_avail"], info["post_samples"])
```

## 快速开始

```bash
# M1 采集层端到端验证（仿真信号 → 触发 → 切片落盘 + 性能基准）
python 07_测试工具/verify_acquisition.py

# 回归测试（09 采集层 21 用例）
python -m pytest 06_测试与验证/test_acquisition.py -q
```

## 与 01 预处理的关系（重要）

v2.1 的流式状态延续滤波（慢基线 + 带通）保证快路径帧**保留基波**；看门狗/触发
依赖**绝对幅值**，因此预处理必须 `norm_enabled=False`（幅值保持）。归一化会抹掉
RMS 绝对水平，导致堵转/负载异常判据失准 —— M1 主循环一律使用幅值保持帧。

## 测试状态

`06_测试与验证/test_acquisition.py`：21 用例（环形缓冲 8 / 看门狗 4 / 触发 4 /
切片 2 / 引擎端到端 3）。全量回归 84/84 见仓库 README。
