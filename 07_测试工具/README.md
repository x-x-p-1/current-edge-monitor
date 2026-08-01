# 07 — 测试工具模块

## 模块概述

集中存放**开发/验证用的工具脚本**（非正式 pytest 套件，正式套件在 `06_测试与验证/`）。
这些工具用于：仿真数据自检、各层 v2 功能验证、M0 端到端仿真、实时看板生成。

```
07_测试工具/
├── verify_simulator.py       # 数据发生器数值自检（RMS/相位/堵转/边带）
├── verify_preprocess.py      # 预处理 v2：三相 (N,3)、频带保留、流式
├── verify_features.py        # 特征 v2：快/慢路径、三相对称分量、MCSA 边带
├── verify_models.py          # 过程状态机：负载周期 / 堵转 / 停机
├── verify_backend.py         # 05 推理后端工厂与错误处理
├── run_m0_simulation.py      # M0 端到端仿真（文本报告 + PNG）
└── export_dashboard.py       # 生成实时可视化看板 sim/dashboard.html
```

## 运行

```bash
# 从算法框架根目录运行
python 07_测试工具/verify_simulator.py
python 07_测试工具/verify_preprocess.py
python 07_测试工具/verify_features.py
python 07_测试工具/verify_models.py
python 07_测试工具/verify_backend.py

# M0 端到端仿真 → sim/m0_report_utf8.txt + sim/m0_simulation.png
python 07_测试工具/run_m0_simulation.py

# 实时看板 → sim/dashboard.html（浏览器打开，连续循环播放）
python 07_测试工具/export_dashboard.py
```

## 与正式测试的分工

| 位置 | 内容 | 定位 |
|------|------|------|
| `06_测试与验证/` | `test_*.py`（pytest/unittest） | 正式回归套件，CI 用，全量 57/57 |
| `07_测试工具/` | 上述脚本 | 开发期功能验证 / 仿真 / 可视化，带打印输出可人工观察 |

> 所有脚本都以 `算法框架` 根目录为工作目录运行（`sim/` 输出落在根目录下）。
