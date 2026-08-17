# rishvanjay/MCSA — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/rishvanjay/MCSA
> **技术栈**：Python（FFT/Welch/Hilbert/SVM/回归/TensorFlow 脚本集，17★，无 release）
> **定位**：MCSA ML 脚本集（研究生项目风格实验代码）
> **契合度**：★ 低（方法枚举 + 实验数据备查）

## 一、仓库概览

README 只有一句 "Motor Current Signature Analysis ML Scripts"。顶层散放一批 Python 脚本：`FFT.py`、`bandpass.py`、`hilbert.py`、`directCurrent.py`、`freqAmp.py`、`prestigeWelch.py`（Welch PSD）+ ML 分类脚本 `SVM.py`、`regression.py`、`tensorflow.py`、`test_model.py`，配 `client.py/server.py`（上下位机通信），多批实验数据目录（`EM faulty`、`EM_04062018` 等）。单作者，未维护，无文档，含 `.pyc` 编译产物。

## 二、值得借鉴（为什么）

### 1. MCSA + ML 方法组合（枚举参考）
- 借鉴：SVM / 回归 / TensorFlow 做故障分类的组合。
- 为什么：可作为"用 ML 替代/辅助阈值判定"的方法枚举（我们若在 M2 用 ML 分类，可参考其方法面）。

### 2. 实验数据目录
- 借鉴：不同日期、含 faulty 的 EM 实验数据。
- 为什么：可能作为算法验证的原始数据来源（需自行评估质量）。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| 工程组织混乱 | 脚本堆根目录、无文档、含编译产物、未维护 |
| 无预处理规范/验证流程 | 无法保证可复现 |
| 工程价值 | 被 mcp-server-mcsa 完全覆盖 |

## 四、结论

**借鉴度 ★ 低**。仅作"MCSA+ML 方法集 + 实验数据"的备查，不入深入学习。工程参考价值被 mcp-server-mcsa 覆盖。
列入不推荐关注。
