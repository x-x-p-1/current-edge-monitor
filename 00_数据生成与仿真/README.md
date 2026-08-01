# 00 — 数据生成与仿真模块

## 模块概述

在真实采集硬件就绪前，按《采集侧确认清单》的**原始数据契约**生成合成三相电流数据，
用于管线自检、采集对照基准与 ML 打底。

```
合成三相电流(正常/故障) ──▶ 预处理 → 特征 → 触发 → 切片（管线自检）
                        ──▶ 真实数据对照体检（SNR/同步性/连续性）
                        ──▶ 带标签合成故障样本（ML 正常建模/异常检测）
```

**设计依据**：西门子 LGF 波形发生算法族（Sinus/SawTooth/Rectangle）的边缘管线自检理念；
三菱 MCSA 边带模型（转子条故障边带 f1 ± 2·s·f1）。

## 文件结构

| 文件 | 功能 |
|------|------|
| `current_simulator.py` | 三相电流原始数据仿真器（正常 + 8 类故障注入） |

> 仿真器的**自检 / M0 仿真 / 看板生成**等工具已移至 `07_测试工具/`（见该模块 README）。

## 快速开始

```bash
# 仿真器数值自检（工具已移至 07_测试工具）
python 07_测试工具/verify_simulator.py

# M0 端到端仿真
python 07_测试工具/run_m0_simulation.py

# 实时看板
python 07_测试工具/export_dashboard.py
```

```python
from current_simulator import generate_dataset, Fault

# 正常三相电流，带负载周期（启动/空载/负载）
sig, ts, meta = generate_dataset(
    fs=16000, duration=10.0, f1=50.0,
    load_profile=[(0.0, 2.0, 0.3), (2.0, 8.0, 1.0), (8.0, 10.0, 0.3)],
)

# 注入转子条边带故障（MCSA 特征）
sig2, ts2, meta2 = generate_dataset(
    fs=16000, duration=10.0, f1=50.0,
    faults=[Fault(kind="rotor_sideband", start=5.0, dur=3.0, slip=0.03, depth=0.02)],
)
```

## 支持参数（对齐原始数据契约）

| 参数 | 默认 | 说明 |
|------|------|------|
| `fs` | 16000 | 采样率 (SPS/通道) |
| `duration` | 10.0 | 时长 (s) |
| `f1` | 50.0 | 基波频率（变频器输出频率） |
| `amplitude` | 1.0 | 基波幅值（标幺） |
| `harmonics` | {5,7,11,13} | 电机特征谐波 |
| `fsw` / `pwm_depth` | 8000 / 0.03 | 变频 PWM 纹波 |
| `snr_db` | 60 | 信噪比（关联位深/ENOB） |
| `adc_bits` / `full_scale` | 24 / 2.0 | 量化与满量程 |
| `imbalance` | 0.0 | 三相不平衡度 |
| `load_profile` | None | 负载包络 [(t0,t1,amp), ...] |
| `faults` | [] | 故障注入列表 |

## 支持故障类型（8 类，带标签）

| kind | 说明 | 关键参数 |
|------|------|----------|
| `stall` | 堵转（幅值突增） | depth |
| `load_step` | 负载突变 | depth |
| `rotor_sideband` | MCSA 转子条边带 | slip, k, depth |
| `spike` | 随机冲击脉冲 | density, depth |
| `harmonic` | 谐波畸变升高 | depth |
| `unbalance` | 三相不平衡 | depth |
| `degradation` | 全程缓变劣化（慢性） | depth |
| `arc` | 电弧（零休+高频爆发） | depth |

## 输出

- `signal`: `(N, 3)` float，三相电流（安培/标幺）
- `timestamps_ns`: `(N,)` int64，模拟 UTC 时戳
- `metadata`: dict，含契约参数 + 削波比例 + 故障标签

## 红线提醒

合成数据用于**管线自检 / QA 基准 / ML 预训练**，模型最终验收必须以真实采集数据为准。
