# 13115827885/PFLD 疲劳驾驶检测 — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/13115827885/Research-on-Fatigue-Driving-Detection-System-Based-on-PFLD-Face-Recognition
> **技术栈**：Python + PyTorch + MTCNN + PFLD + ONNX + TensorFlow/TFLite + OpenCV + Kivy/PyInstaller
> **定位**：基于 PFLD 人脸关键点的疲劳驾驶检测系统（边缘实时 + 滑窗时序融合降误报）
> **契合度**：★★ 中（降误报方法论 + 边缘部署管线）

## 一、仓库概览

实时驾驶员疲劳检测系统，运行于边缘设备。MTCNN 人脸检测 + PFLD 98 点关键点定位，计算 **EAR（眼部纵横比）/ MAR（嘴部纵横比）/ PERCLOS（闭眼百分比）** 疲劳指标；**滑窗统计的时序融合**降误报；检测到疲劳触发语音告警。边缘 28 FPS，疲劳 F1=0.92。

关键实现（README 核实）：
- **MTCNN** 三级级联（P-Net/R-Net/O-Net）人脸检测
- **PFLD** 轻量关键点（MobileNet 骨干 + 多尺度融合 + 角度加权损失，强鲁棒头姿）
- **EAR/MAR/PERCLOS** 疲劳指标（EAR<0.25 闭眼；MAR>0.5 打哈欠；PERCLOS>30%/10s 疲劳告警）
- **时序融合**：滑窗存 EAR + 连续帧计数器（连续 5 帧低 EAR/高 MAR 触发）+ PERCLOS 累积
- **边缘部署管线**：PyTorch(.pth) → ONNX → TensorFlow SavedModel → TFLite(.tflite)
- Kivy 桌面应用 + PyInstaller 打包（Windows）

性能：WFLW 测试集 NME 0.0834；实时 ~28 FPS；疲劳 F1 0.92。

## 二、值得借鉴（为什么）

### 1. 滑窗统计 + 连续帧计数 + 累积率的时序融合降误报（方法论直接相关）
- 借鉴：滑窗存 EAR 指标 + 连续帧计数器（5 帧确认）+ PERCLOS 累积率（30%/10s 不可逆告警）。
- 为什么：与我们触发引擎的**平滑 + 确认去抖 + 累积度量**完全同构——"短窗确认（防瞬时尖峰）+ 累积窗判定（防漏检）"的双层时序策略，可对照我们 TriggerEngine 的 confirm_debounce 与累积度量设计。

### 2. 多级告警状态机（绿/黄/红）
- 借鉴：Awake（绿）→ Warning（黄，连续帧）→ Fatigue（红，PERCLOS）+ 音频告警。
- 为什么：对应我们"正常 → 预警 → 告警"的分级输出；分级的阈值递进设计可参考。

### 3. PyTorch → ONNX → TFLite 边缘部署管线
- 借鉴：多级模型转换管线，TFLite 无 GPU 边缘推理。
- 为什么：我们 M3 若引入轻量模型，PyTorch → ONNX → 板端（RKNN/TFLite）的转换链是标准路径，可参考其转换脚本组织。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| 人脸检测/关键点（MTCNN/PFLD） | 视觉域，与电流诊断无关 |
| Kivy 桌面 UI / PyInstaller | 我们看板是 HTML，非桌面应用 |
| WFLW 数据集/训练 | 视觉数据集 |
| 具体阈值（EAR/MAR） | 领域特定，仅"时序融合结构"可借鉴 |

## 四、结论

**借鉴度 ★★ 中**。它是"边缘实时检测 + 时序融合降误报"的完整研究样例，与我们交集在**时序融合与告警策略**：
① 滑窗 + 连续帧确认 + 累积率的三层时序融合（TriggerEngine 的直接对照）；
② 绿/黄/红多级告警状态机；
③ PyTorch→ONNX→TFLite 边缘部署管线（M3 备查）。
人脸算法不借鉴。建议把"确认 + 累积"双层时序策略纳入 03_检测模型 / 04_决策 的设计参考。
