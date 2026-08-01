# 02 — 特征提取模块手册

## 模块概述

特征提取是算法栈的**第二层**，将预处理后的干净时域波形压缩为低维、具有明确物理含义的特征向量，供检测模型使用。

```
干净波形 → 时域特征 + 频域特征 + 时频域特征 + 统计特征 + 三相特征 → 特征向量
```

## v2 变更记录（对齐三相变频电机方向）

| 变更 | v1 | v2 |
|------|----|----|
| **快/慢路径分窗** | 单一 5.12ms 窗塞进所有特征（频域物理不成立） | **拆分快路径（短窗 ms，时域+包络）与慢路径（长窗 s，频域+MCSA+三相）**，见 `cadence.py` |
| **三相跨相特征** | 无 | **新增 `three_phase.py`**：幅值平衡/相角偏差/对称分量/三相相关 |
| **MCSA 边带** | 无 | **新增 `compute_sideband_energy`**（转子条 f1±2·s·f1） |
| 编排入口 | 各域独立 `extract_*_features` | 新增 `extract_fast_features` / `extract_slow_features` 编排器 |

> 各域原有的 `extract_*_features`（时域/频域/时频/统计）保持兼容，可单独调用；
> 快/慢路径编排器在它们之上做多节拍组织。

**核心设计哲学**（借鉴西门子 CMS 三维指标体系）：

| 西门子原设计（振动域） | 电流域迁移 |
|---|---|
| vRMS（10Hz–1kHz） | I_rms（信号总能量） |
| aRMS（1kHz–10kHz） | 高频能量比（>2kHz） |
| DKW 无量纲自适应基线 | 归一化特征 + 基线学习 |
| Bands Allocation 频段划分 | 电流频谱 5 频段能量 |
| 希尔伯特包络解调 | 电流包络统计特征 |

---

## 文件结构

| 文件 | 域 | 特征数 | 核心输出 |
|------|-----|--------|----------|
| `time_domain.py` | 时域 | 11 个 | RMS、峰值因子、峭度、过零率、零休时间 |
| `frequency_domain.py` | 频域 | 20+ 个 | 谐波幅值、THD、高频能量比、频段能量 |
| `time_frequency.py` | 时频域 | 14 个 | DWT 能量/熵、包络统计 |
| `statistical.py` | 统计 | 13 个 | 箱线图、四分位偏度、离群比例、趋势斜率 |

---

## 时域特征详解

### 特征清单

| 特征 | 公式 | 物理意义 | 电弧灵敏度 |
|------|------|----------|-----------|
| **RMS** | $\sqrt{\frac{1}{N}\sum i_k^2}$ | 电流热效应 | ⭐ |
| **峰值因子** | $\max|i| / I_{rms}$ | 波形尖峰程度 | ⭐⭐⭐ |
| **波形因子** | $I_{rms} / I_{avg}$ | 偏离正弦度 | ⭐⭐ |
| **峭度** | $E[(i-\mu)^4] / \sigma^4 - 3$ | 脉冲型异常 | ⭐⭐⭐ |
| **偏度** | $E[(i-\mu)^3] / \sigma^3$ | 波形不对称 | ⭐⭐ |
| **过零率** | 过零次数 / N | 平肩部特征 | ⭐⭐⭐ |
| **ΔI 统计** | ΔI 均值/方差/最大 | 瞬态突变 | ⭐⭐ |
| **零休时间** | $T_{|i|<tol}$ | 电弧重燃间隙 | ⭐⭐⭐ |

### 电弧检测核心特征：峰值因子 + 过零率 + 零休时间

这三种时域特征直接对应 GB 14287.4 标准中定义的串联电弧三大特征：

1. **峰值因子飙升** → 电弧电流充满尖峰毛刺
2. **过零率下降** → 过零附近出现"平肩部"
3. **零休时间延长** → 电弧在零点熄灭到重燃的时间间隙

---

## 频域特征详解

### 频谱生成

```
信号 → 去均值 → 汉宁窗 → FFT → 单边幅度谱/功率谱
```

- **窗函数**：汉宁窗 $w[k] = 0.5(1 - \cos(2\pi k/(N-1)))$
- **频率分辨率**：$\Delta f = f_s / N$

### 频段能量划分（Band Energy）

借鉴西门子 Bands Allocation，将全频段划分为 5 个子带：

| Band | 频段 | 对应物理来源 |
|------|------|-------------|
| Band 1 | 45–55 Hz | 基波能量 |
| Band 2 | 100–350 Hz | 3/5/7 次谐波 |
| Band 3 | 500–2000 Hz | 开关噪声、低次间谐波 |
| Band 4 | 2000–5000 Hz | **电弧高频特征区** |
| Band 5 | 5000–10000 Hz | 超高频 EMI |

$$\text{BandEnergy}_k = \sqrt{\sum_{f=f_{start}}^{f_{end}} |A(f)|^2}$$

### 高频能量比

$$\text{HF Ratio} = \frac{\sum_{f>2000\text{Hz}} |A(f)|^2}{\sum_{\text{all }f} |A(f)|^2}$$

- 正常阻性负载：HF Ratio ≈ 0
- 电弧故障：HF Ratio >> 0

### THD 总谐波失真

$$\text{THD} = \frac{\sqrt{\sum_{h=2}^{H} I_h^2}}{I_1} \times 100\%$$

| THD 范围 | 负载类型推测 |
|----------|------------|
| < 5% | 纯阻性负载 |
| 5–20% | 开关电源、LED 灯 |
| 20–50% | 变频器、整流电路 |
| > 50% | 电弧故障（严重畸变） |

---

## 时频域特征详解

### 离散小波变换 (DWT)

4 级 `db4` 小波分解的频段划分（50kSPS）：

| 系数 | 频段 | 电弧敏感度 |
|------|------|-----------|
| cD1 | 12.5–25 kHz | ⭐ |
| cD2 | 6.25–12.5 kHz | ⭐⭐ |
| **cD3** | **3.125–6.25 kHz** | **⭐⭐⭐** |
| cD4 | 1.56–3.125 kHz | ⭐⭐ |
| cA4 | 0–1.56 kHz | ⭐ |

cD3 频段恰好对应电弧的高频噪声集中区域。

### 希尔伯特包络

借鉴西门子 SM1281 的硬件包络解调三步法，在电流域实现：

1. **带通提纯** → 2. **希尔伯特变换** $z(t) = x(t) + j\mathcal{H}\{x(t)\}$ → 3. **包络提取** $e(t) = |z(t)|$

包络统计特征（均值/标准差/变异系数）可反映电弧电流的不规则脉动。

---

## 统计特征详解

### 箱线图五数概括

借鉴西门子 `LGF_Boxplot` 的稳健统计方法：

$$\text{Summary} = \{I_{min}, q_{25}, q_{50}, q_{75}, I_{max}, \text{IQR}\}$$

### 四分位偏度

$$\text{Skew}_{quartile} = \frac{q_{75} + q_{25} - 2q_{50}}{q_{75} - q_{25}}$$

相比三阶矩偏度，四分位偏度对电弧电流中的突变脉冲**极度鲁棒**。

### IQR 离群点检测

借鉴西门子 Outlier Detection：

$$\text{Outlier} \iff x \notin [q_{25} - 1.5\cdot\text{IQR},\; q_{75} + 1.5\cdot\text{IQR}]$$

离群点比例直接量化电弧波形的毛刺密度。

### 趋势斜率

借鉴西门子 `LGF_RegressionLine` 最小二乘回归：

$$m = \frac{n\sum(xy) - \sum x\sum y}{n\sum x^2 - (\sum x)^2}$$

监控特征值（RMS、THD、离群比例）随时间单调上升 → 劣化预警。

---

## 完整用法

```python
from 02_特征提取.time_domain import extract_time_domain_features
from 02_特征提取.frequency_domain import extract_frequency_domain_features
from 02_特征提取.time_frequency import extract_time_frequency_features
from 02_特征提取.statistical import extract_statistical_features

# 从预处理后的信号提取特征
td_feats = extract_time_domain_features(clean_signal, sample_rate=50000)
fd_feats = extract_frequency_domain_features(clean_signal, sample_rate=50000)
tfd_feats = extract_time_frequency_features(clean_signal)
stat_feats = extract_statistical_features(clean_signal)

# 转为 numpy 数组（可直接送入 ML 模型）
td_vec = td_feats.to_array()    # shape (11,)
fd_vec = fd_feats.to_array()    # shape (20,)
tfd_vec = tfd_feats.to_array()  # shape (14,)
stat_vec = stat_feats.to_array() # shape (13,)

# 合并为完整特征向量
full_features = np.concatenate([td_vec, fd_vec, tfd_vec, stat_vec])  # shape (58,)
```

---

## 特征选择建议

| 检测任务 | 推荐特征组合 |
|----------|------------|
| **电弧检测** | 峰值因子 + 过零率 + 零休时间 + 高频能量比 + cD3能量 + 离群比例 |
| **负载识别** | THD + 波形因子 + 谐波3/5/7幅值 + 频谱质心 |
| **异常检测** | 全量 58 维特征 → AutoEncoder 压缩 |
| **劣化评估** | 趋势斜率 + RMS + 离群比例时间序列 |

---

## 参考标准

- GB 14287.4-2014 附录A — 电弧时域/频域特征
- IEEE 519 — 谐波限值
- 西门子 LGF — 统计学算法族
- 西门子 CMS — Bands Allocation & DKW
