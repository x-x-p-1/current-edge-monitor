# apache/streampipes — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/apache/streampipes
> **技术栈**：Java + TypeScript + NATS/MQTT/Kafka/Pulsar + 时序库 + Apache-2.0（736★ / 239 fork）
> **定位**：Apache 工业 IoT 数据平台（端到端流处理 + 可视化管道 + 时序探索）
> **契合度**：★★★ 高（流处理管道元素 + 变化检测处理器）

## 一、仓库概览

Apache StreamPipes 是 Apache 基金会的开源工业 IoT 数据平台：连接工业数据源、构建实时流处理管道、探索时序数据、交付实时运营洞察。定位"工业 IoT 数据工具箱"，面向非技术用户的可视化 + 面向开发者的扩展框架。

核心能力（README 核实）：
- **Connect**：连接工业系统（OPC UA 深度支持——web 节点浏览器 + OPC UA 事件、PLC、MQTT、REST、Pulsar、Kafka）
- **Pipeline Editor**：可视化流分析管道（适配器/处理器/汇）
- **Charts & Dashboards**：时序图表 + 实时看板
- **Asset Management**：资产中心化组织
- 标准安装自带消息系统（NATS 默认）+ 时序库
- 企业级：用户/角色管理、OAuth 2.0、地理分布式部署（OT/IT 网络间）
- 扩展：Java SDK 构建自定义 adapter/processor/sink；Python/Go/Java 客户端；REST API

仓库结构：`streampipes-connect-*`（适配器）、`streampipes-processors-*`（处理器：change-detection / statistics / aggregation / pattern-detection / filters / transformation）、`streampipes-sinks-*`（汇：数据库/代理/通知）、`streampipes-sdk`、`ui`、`installer`（Compose/CLI/k8s）。

## 二、值得借鉴（为什么）

### 1. Pipeline Element（adapter/processor/sink）微服务扩展模型
- 借鉴：每个管道元素是独立可部署的微服务，**可在中心或靠近边缘部署**（取决于延迟与基础设施约束）。
- 为什么：与我们的「采集驱动抽象 + 特征处理 + 落库/上报」同构——"一个功能一个可独立部署单元"的拆分方式，正是 M1 采集层 + 后续特征管线的组织参照。

### 2. 现成处理器库（变化检测/统计/聚合/模式检测）
- 借鉴：`change-detection`、`statistics`、`aggregation`、`pattern-detection` 等 JVM 处理器。
- 为什么：这些正是我们「基线 + 置信区间 + 累积度量」方向的**现成算法参考**——变化检测/统计/聚合的实现可直接对照我们 02/03 模块。

### 3. 多消息层 + 时序存储的一体化预装
- 借鉴：标准安装自带消息系统（NATS 默认，可换 MQTT/Kafka/Pulsar）+ 时序库。
- 为什么：印证「消息总线 + 时序存储」是工业数据平台的标配底座（对应我们 MQTT 上行 + 特征日志存储）。

### 4. OPC UA 深度支持
- 借鉴：web 节点浏览器 + OPC UA 事件订阅。
- 为什么：若 VFD/PLC 走 OPC UA，其节点浏览/事件模型是接入参照。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| 重型 Java 平台本体 | 我们目标 RK3588 轻量 Python 栈；不部署平台 |
| UI（图表/看板/资产管理） | 我们已有自定义 HTML 看板 |
| OAuth 2.0 / 多租户 / 分布式部署 | 单站边缘部署无需企业级 IAM |
| 流处理框架（Flink 等 wrapper） | 我们实时管线是轻量 Python，不需要分布式流引擎 |

## 四、结论

**借鉴度 ★★★ 高**。它是工业流处理的标准件，对我们的价值在**管道元素扩展模型**（一个功能一个独立部署单元，可跑在边缘）和**现成处理器库**（变化检测/统计/聚合/模式检测，直接对照我们 02/03 模块）。
不部署平台，借鉴组织方式与算法处理器。建议纳入英文批次核心参考。
