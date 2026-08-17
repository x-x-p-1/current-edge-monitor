# iweidujiang/java-industrial-smart — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/iweidujiang/java-industrial-smart
> **技术栈**：Java/Spring Boot + MyBatis-Plus + Redis + MySQL + Vue3/ECharts + Three.js
> **定位**：Java 工业智能专栏配套代码（PLC 接入 → 时序缓存 → 数据预判 → 监控大屏 → 数字孪生）

## 一、仓库概览

一个以"从 Modbus 到工业智能"为主题的 Java 专栏项目，按模块分目录：

```
06-plc-unified-adapter    PLC 统一接入（PlcProtocolAdapter 接口：Siemens S7 / Modbus 等）
07-data-cache-persistence 工业时序数据缓存与持久化（Redis 实时 + MySQL 落库）
08-data-prediction        数据预判（核心：历史阈值 + 趋势判断）
09-industrial-monitor     工业监控大屏（Vue3 + ECharts，温度/压力曲线 + 告警弹窗）
11/12-digital-twin        数字孪生（Three.js 3D 场景 + 空间定位热图）
```

最相关的是 **08-data-prediction**：`PredictionService` 实现"历史阈值 + 趋势判断"的实时预判。

## 二、值得借鉴（为什么）

### 1. 数据预判流程骨架 —— 与我们的「基线 + 置信区间 + 累积度量」同源
对应我们已定方向（基线向量 + 马氏置信区间 + CUSUM 累积），它是**单特征简化版的可运行参照**。
- 借鉴流程（从 `PredictionService` 核实）：
  ```
  统计历史 → 算 historyAvg/Min/Max（基线阈值）
       → 实时判定 predictionStatus(0 正常 / 1 预警 / 2 异常)
       → 批量持久化预判结果
       → 定时任务周期重算阈值（PredictionScheduledTask）
  ```
- 为什么：这条"算基线 → 实时判 → 落库 → 周期更新基线"的骨架和我们 M1 收尾
  （特征日志常存 + 预判结果落库）完全一致，可直接映射；我们只需把单特征阈值升级为
  多特征马氏 + CUSUM，流程不动。

### 2. 阈值配置化（PredictionThresholdConfig）
对应我们把预测/触发阈值放进 `config.yaml`。
- 借鉴：所有阈值独立配置类，改参数不动代码。
- 为什么：边缘设备现场调参频繁，配置化是基本要求。

### 3. 统一 PLC 接入层（PlcProtocolAdapter）
与 BetterIOT 的 IDrive 同理（遥测接入抽象），这里多一个参考实现（Java 版）。
- 为什么：将来接 VFD/PLC 工况信号时，三份仓库（BetterIOT/Java/LotusBridge）给出三种语言的
  同一套"接口 + 实现 + 注册"模式，说明这是行业共识，可放心采用。

### 4. Redis 实时缓存 + 时序库分层的存储
对应特征日志常存的"热/冷"分层。
- 借鉴：最新值进 Redis（毫秒级读），历史批量进 MySQL/时序库。
- 为什么：我们特征日志也需要"最近值热存（看板/触发）+ 历史归档"两层，避免单库压力。

### 5. 告警触发 + 前端弹窗（AlertToast）
对应报警输出/看板。
- 借鉴：后端阈值告警 → 前端 toast 轮询展示。
- 为什么：M1 之后需要把触发事件可视化，这套前后端交互是现成范式。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| Java/Spring Boot 全家桶 | JVM 在边缘板开销大，且与 Python 算法栈不一致；仅借鉴业务逻辑 |
| 单特征简单阈值预判（如温度>60 直接告警） | 我们做多特征向量基线 + 马氏 + 累积，是其严格升级版，无需退化模仿 |
| MockDataGenerator（正弦+随机模拟数据） | 我们已有 00 物理仿真器（真实三相 + 故障注入），比它强得多 |
| 数字孪生/3D（Three.js） | 与核心电流诊断无关，量产可视化阶段再议 |

## 四、结论

**借鉴度 ★★★ 高**。最有价值的是 **08 数据预判的流程骨架**——它和你们
「基线 + 置信区间 + 累积度量」方向同源，是"先做特征日志 + 预判落库"时最直接的参考实现。
其余（阈值配置化 / 遥测接入 / 分层存储）与 BetterIOT 的借鉴点互相印证。
