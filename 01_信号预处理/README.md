# 01 — 信号预处理模块手册

## 模块概述

信号预处理是算法栈的**第一层**，负责将原始 ADC 采样数据转化为干净、规范、可直接用于特征提取和模型推理的信号。支持**单相 `(N,)` 与三相 `(N, C)`（时间×通道）**数据，并提供**流式处理**（快路径，ms 级）。

```
原始采样(N,/N,3) → 直流偏置去除 → 带通滤波 → 幅值归一化 → [相位对齐] → 干净信号
```

## v2 变更记录（对齐三相变频电机方向）

| 变更 | v1 | v2 |
|------|----|----|
| 通道结构 | 仅单通道 (N,)，二维按 (batch, N) | **统一 (N, C) 三相逐通道处理**，1D 完全兼容 |
| 带通参数 | 45Hz~2500Hz（市电/电弧） | **lowcut=5Hz / highcut=4000Hz**，可配置 |
| 带通矛盾 | 2500Hz 把 >2.5kHz 诊断频带滤掉，与高频特征自相矛盾 | **修掉**：highcut 不得低于诊断频带上限 |
| 流式处理 | TODO（未实现） | **已实现** `process_streaming`：环形缓冲 + 滑动窗口 |
| 默认采样率 | 50000 | **16000**（对齐数据契约） |

**核心设计哲学**（借鉴西门子 SM1281 AFE 方法论）：

| 西门子原设计（振动域） | 电流域迁移 |
|---|---|
| 高通 RC 耦合阻断 DC 偏置 | 滑动均值减法去直流 |
| 带通 10Hz–1kHz（速度量级） | 带通 45Hz–2.5kHz（工频 + 电弧高频） |
| 24-bit ΔΣ ADC 过采样 | 外部高速 ADC + DMA |
| DSP 硬件滤波流水线 | CPU A55 小核预处理 |

---

## 文件结构

| 文件 | 功能 | 核心算法 |
|------|------|----------|
| `filters.py` | 数字滤波器 | 滑动均值去DC、巴特沃斯带通、移动平均、SG滤波、陷波 |
| `normalization.py` | 归一化 | Z-Score / MinMax / RMS / Peak 归一化 |
| `alignment.py` | 相位对齐 | 过零点检测、窗口对齐、整周期提取 |
| `preprocess.py` | 预处理管线 | 串联上述步骤的 Pipeline 主控 |

---

## 核心算法详解

### 1. 直流偏置去除 (`remove_dc_offset`)

#### 数学原理

$$I_{ac}[k] = I_{raw}[k] - \frac{1}{W}\sum_{i=k-W+1}^{k} I_{raw}[i]$$

滑动窗口估计局部 DC 分量，从原始信号中减去。

#### 工程意义

ADC 的零点漂移和传感器静态偏置会严重干扰后续特征计算：
- RMS 被抬高 → 负载判断失准
- FFT 出现 DC 分量 → 频谱泄露加剧
- 积分运算发散

```python
from 01_信号预处理.filters import remove_dc_offset
clean = remove_dc_offset(raw_signal, window=100)
```

---

### 2. 巴特沃斯带通滤波 (`butter_bandpass_coefficients` + `bandpass_filter`)

#### 频段选择（VFD 三相电机方向，可配置）

| 频段 | 截止频率 | 作用 |
|------|---------|------|
| 低频截止 | **5 Hz**（默认） | 阻断 <5Hz 的非周期低频漂移；**必须低于 VFD 最低输出频率**，否则低速基频被杀 |
| 高频截止 | **4000 Hz**（默认） | 滤除带外高频 EMI；**不得低于诊断频带上限**（MCSA 边带/谐波/包络频带） |
| 保留频段 | **5–4000 Hz** | 变频基频 + 谐波 + MCSA 边带 + 包络高频带 |
| PWM 纹波 | 由模拟抗混叠处理 | 数字带通不在此一刀切（fs=16k 时 fsw 靠近 Nyquist） |

> ⚠️ 具体 lowcut/highcut 应根据《电气工程师需求清单》#7 的变频器开关频率 fsw 与被测电机最低运行频率确定。

#### 数学原理

4 阶巴特沃斯 IIR，零相位滤波（`filtfilt`）：

$$H(s) = \frac{1}{\sqrt{1 + (s/\omega_c)^{2n}}}, \quad n=4$$

#### 零相位的重要性

使用 `scipy.signal.filtfilt`（前向+反向滤波）确保**不引入任何相位偏移**。相位偏移会导致过零点位置错位，扭曲电弧"平肩部"特征。

```python
b, a = butter_bandpass_coefficients(lowcut=45, highcut=2500, fs=50000)
filtered = bandpass_filter(signal, b, a)
```

---

### 3. 信号归一化 (`normalize_signal`)

#### 四种方法

| 方法 | 公式 | 输出范围 | 适用场景 |
|------|------|---------|----------|
| **Z-Score** | $(x-\mu)/\sigma$ | ~N(0,1) | ⭐ 深度学习模型（推荐） |
| Min-Max | $(x-x_{min})/(x_{max}-x_{min})$ | [0,1] | 传统 ML |
| RMS | $x/I_{rms}$ | 与幅值无关 | 波形形态分析 |
| Peak | $x/\max(|x|)$ | [-1,1] | 过零特征保留 |

#### 工程意义（借鉴 DKW 思想）

归一化使模型关注**波形形态（Shape）**而非**绝对幅值（Amplitude）**，实现对不同额定电流负载的通用检测。

```python
# 训练时计算参数
params = compute_normalization_params(train_data, method="zscore")  # {"mu": 0.1, "sigma": 0.5}

# 推理时复用参数
normalized = apply_normalization_params(signal, params, method="zscore")
```

---

### 4. 相位对齐 (`align_to_zero_crossing`)

#### 过零点检测

寻找满足以下条件的采样点：

$$\text{signal}[k-1] < -\text{tolerance} \quad \text{AND} \quad \text{signal}[k] \geq +\text{tolerance}$$

#### 亚采样精度插值

$$x_{zero} = k_{before} + \frac{0 - y_{before}}{y_{after} - y_{before}}$$

#### 电弧检测意义

电弧电流波形在过零点附近出现**"平肩部"（Shoulder）**——电流在零附近维持一小段时间。窗口对齐到过零点确保这一关键特征不被窗口截断。

```python
aligned = align_to_zero_crossing(signal, target_length=256, direction="positive")
cycle = extract_full_cycle(signal, sample_rate=50000, nominal_freq=50)
```

---

## Pipeline 完整用法

```python
from 01_信号预处理.preprocess import CurrentPreprocessor, PreprocessConfig

# 三相 16kSPS 数据契约（对齐《采集侧确认清单》）
config = PreprocessConfig(
    sample_rate=16000.0,
    channels=3,
    window_size=256,
    stride=128,
    dc_removal_enabled=True,
    dc_window=100,
    filter_enabled=True,
    filter_lowcut=5.0,      # 低于 VFD 最低输出频率
    filter_highcut=4000.0,  # 不低于诊断频带上限
    norm_enabled=True,
    norm_method="zscore",
)

pp = CurrentPreprocessor(config, sample_rate=16000.0)

# 批处理：三相 (N, 3) → (N, 3)
clean_abc = pp.process(raw_abc)

# 流式处理（快路径，ms 级）：送入一批新采样，满窗口出帧
for chunk in raw_stream_chunks:
    frames = pp.process_streaming(chunk)   # [] 或 [processed_frame, ...]
    for frame in frames:
        feed_fast_path(frame)
```

# 逐帧处理
for raw_frame in adc_stream:
    clean_signal = preprocessor.process(raw_frame)
    # → 送入特征提取 / 模型推理
```

---

## 配置参数参考

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `dc_removal.window` | 100 | DC 估计滑动窗口 |
| `bandpass_filter.order` | 4 | 巴特沃斯阶数 |
| `bandpass_filter.lowcut` | 45 Hz | 低频截止 |
| `bandpass_filter.highcut` | 2500 Hz | 高频截止 |
| `normalization.method` | zscore | 归一化方法 |

---

## 性能指标

| 操作 | 256点耗时 | 说明 |
|------|----------|------|
| DC 去除 | ~5 µs | 纯向量运算 |
| 带通滤波 | ~50 µs | filtfilt(前向+反向) |
| Z-Score 归一化 | ~3 µs | 纯向量运算 |
| 过零检测 | ~5 µs | 条件扫描 |
| **Pipeline 总计** | **~70 µs** | 远低于 2.56ms 帧间隔 |

---

## 参考标准

- GB 14287.4-2014 附录A — 电弧特征参数
- 西门子 SM1281 AFE — 信号调理方法论
- IEC 61000-4-7 — 谐波测量窗函数建议
