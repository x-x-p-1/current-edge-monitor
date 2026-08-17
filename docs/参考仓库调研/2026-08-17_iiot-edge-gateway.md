# KuzinHouse/IIoT-Edge-Gateway — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/KuzinHouse/IIoT-Edge-Gateway
> **技术栈**：TypeScript + Next.js 16 + Tailwind + Prisma/SQLite + WebSocket + MIT（3★，俄语文档）
> **定位**：EMQX Neuron 级别的工业 IoT 边缘网关（南向设备/北向应用/管道/告警，Web HMI）
> **契合度**：★★ 中（协议驱动注册表 + 设备模板含变频器 + 标签告警配置）

## 一、仓库概览

IIoT Edge Gateway 是面向工业 IoT 的边缘网关平台，遵循 EMQX Neuron 架构 + OPA-S（Open Platform Architecture — South）规范管理协议驱动。Next.js 构建的 Web HMI/SCADA 管理南向设备（PLC/传感器/执行器）与北向应用（云/历史库/消息代理）。

核心能力（README 核实）：
- **30+ 协议驱动**：串口（Modbus TCP/RTU/ASCII、HART）、PLC（S7、EtherNet/IP、FINS、MELSEC）、过程自动化（OPC UA、IEC 104、DNP3、IEC 61850）、楼宇（BACnet、KNX）、网络（SNMP）、北向（MQTT v5、Kafka、REST、WebSocket、AWS/Azure）
- **90+ 设备模板 / 12 类**（PLC、**变频器**、传感器、IO 模块、计数器、网关）——选厂商/型号自动得寄存器映射
- **标签管理**：完整生命周期；类型（BOOL/INT16/UINT16/INT32/UINT32/FLOAT32/STRING）、寄存器类型（Holding/Input/Coil/Discrete）、**每个标签告警配置（阈值/死区/延迟）**
- **管道处理**：可视化节点编辑器，13 种节点（南向源/读标签/转换/过滤/聚合/脚本/MQTT 发布/HTTP Push/Kafka/WebSocket/日志/告警检查/延迟）+ 8 个模板
- **北向应用**：MQTT（Mosquitto/EMQX/HiveMQ）、历史库（InfluxDB/TimescaleDB/PI System）、流（Kafka/Kinesis）、云（AWS/Azure/GCP）、REST/WebSocket、SAP、OPC UA Server
- 告警管理：严重级别 + 确认工作流 + 阈值/死区/延迟规则
- 数据模型：**JSON-LD 扁平模型（OPA-S）**，Tag 含 @context/@type/@id/address/dataType/unit/value/quality/timestamp/device
- 诊断：Modbus 测试器、MQTT 测试器、系统健康；Mini-services（Modbus 模拟器 1 万寄存器/WebSocket broker/MQTT bridge）
- 工业 RBAC（L1-L7）、4 级许可、安全（密码策略/IP 白名单/2FA/API 密钥/审计）

## 二、值得借鉴（为什么）

### 1. 设备模板（90+ 含变频器 VFD）
- 借鉴：厂商/型号 → 自动寄存器映射模板。
- 为什么：**变频器模板正是我们 VFD 遥测的参照**——"选型号即得寄存器映射"的设备模板模式可直接映射我们 VFD/PLC 接入。

### 2. 标签级告警配置（阈值/死区/延迟）
- 借鉴：每个标签可配阈值 + 死区 + 延迟。
- 为什么：与我们的触发引擎（EWMA 基线 + K·σ + 确认去抖 + 迟滞释放）思路一致——**"阈值 + 死区 + 延迟"是工业告警的标准配置三元组**，可对照我们确认/迟滞参数设计。

### 3. 协议驱动注册表（30+）
- 借鉴：protocol-registry.ts 集中管理 30+ 驱动定义。
- 为什么：印证"协议驱动注册"模式（同 iot-dc3/StreamPipes），标签数据模型 + 寄存器类型映射是统一遥测接入的参照。

### 4. 管道节点化处理（13 节点 + 8 模板）
- 借鉴：读标签/转换/过滤/聚合/脚本/发布的可视化节点链。
- 为什么：与 StreamPipes 管道元素同构——"采集→变换→聚合→发布"的节点化组织，对应我们数据管线。

### 5. 诊断 + 模拟器内置
- 借鉴：Modbus 测试器 + 1 万寄存器模拟器。
- 为什么：再次印证"仿真/诊断工具内置"（同 iotStudio/StreamPipes）。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| Next.js Web HMI 本体 | 我们看板是自定义 HTML |
| 许可/多租户/RBAC 全套 | 单站边缘场景，过度设计 |
| 4 级许可商业模式 | 非我们关注 |
| 北向多应用（SAP/PI System 等） | 我们只需 MQTT/HTTP 上行 |

## 四、结论

**借鉴度 ★★ 中**。它是"Neuron 级别网关"的 Next.js 实现，工程尚浅（3★、俄语文档），但**设备模板（含变频器）+ 标签告警配置（阈值/死区/延迟）**对我们 VFD 遥测与触发引擎有直接参照价值。
建议作为「遥测接入 + 告警配置」的补充参考（与 EMS-Modbus-Gateway 互补：一个解码表，一个模板+告警）。
