# ForestHubAI/edge-agents — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/ForestHubAI/edge-agents
> **技术栈**：Go 引擎 + TypeScript（React Flow builder）+ OpenAPI contract + Docker（distroless nonroot）+ 多语言（AGPL/双许可，97★ / 32 fork）
> **定位**：30MB 开源边缘 AI agent 运行时（离线默认，GPIO/UART/MQTT 一等公民节点 + 本地 SLM）
> **契合度**：★★ 中（contract-first 架构 + 图工作流 + 离线优先）

## 一、仓库概览

Edge Agents 是 30MB 的边缘 AI agent 运行时，离线运行于 Linux 边缘设备（Raspberry Pi 5、Jetson Orin Nano、STM32MP25、Bosch Rexroth ctrlX CORE）。用可视化方式构建 agent，部署到设备，直接连 GPIO/MQTT/本地 SLM，无需云。

核心特性（README 核实）：
- **工作流引擎**：typed 图运行时，节点含 LLM 调用/硬件 IO/MQTT/web 搜索/内存/控制流；五类边（control/tool/agentTask/agentChoice/agentDelegate）
- **多 provider LLM**：Anthropic/OpenAI/Gemini/Mistral + 本地 SLM（llama.cpp/vLLM/Ollama/OpenAI 兼容端点）
- **硬件一等公民**：GPIO（go-gpiocdev）/ADC/DAC/PWM/UART（go.bug.st/serial）/MQTT（Eclipse Paho）
- **可视化 builder**：React Flow 画布 + CLI（fh-workflow：open/validate/check-schema/update/deploy）
- **contract-first**：`contract/*.yaml`（OpenAPI 3.0.3）单一事实源，Go + TypeScript 从它生成，**CI 防 drift**
- 引擎 headless 无入站 HTTP，单配置文件 `ENGINE_CONFIG_FILE`，独立运行无控制面/账号/入站端口
- 部署：`fh-workflow deploy` 生成自包含 bundle（docker-compose.yml + .env + workflow）；on-device SLM 自动加 llama 组件
- 工业协议（OPC-UA、Modbus）在 roadmap（尚未实现）

## 二、值得借鉴（为什么）

### 1. Contract-first 架构（单一事实源 → 多语言生成）
- 借鉴：OpenAPI schema 为唯一事实源，Go/TS 双端生成，CI 对 schema drift 失败。
- 为什么：与我们数据管线的**类型契约**思想同构——用一份 schema 生成多端绑定，杜绝漂移。我们 06 测试/09 采集层可参考"契约单一来源"的组织方式。

### 2. 图工作流引擎（节点/边/状态机）
- 借鉴：工作流是有向图，引擎按状态机解释（wait event → execute node → transition），节点含 LLM/IO/MQTT/控制流。
- 为什么：我们数据管线（采集→预处理→特征→检测→决策→上报）可看作节点图——其图组织 + 边类型 + 状态机执行是参考。

### 3. 离线默认 + 硬件/MQTT 一等公民
- 借鉴：离线优先设计，GPIO/UART/MQTT 为原生节点而非 REST 垫片。
- 为什么：印证"边缘原生硬件/总线接入优先"理念，与我们"本地零云依赖 + MQTT 总线"一致。

### 4. 可移植的无状态 bundle 部署
- 借鉴：`deploy` 生成自包含部署包（compose + env + config），含 secret 占位。
- 为什么：边缘部署的工程化（配置/凭据/模型分离）参照。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| LLM/agent 运行时本体 | 我们是确定性信号处理，不是 LLM 工作流 |
| 本地 SLM（llama.cpp 等） | 未来 M3 若上轻量模型才相关，非当前 |
| GPIO/UART 硬件节点 | 我们走 ADC/遥测，非 GPIO 控制 |
| AGPL 双许可 | 借鉴思想不搬运代码 |
| 工业协议未实现 | Modbus/OPC UA 在 roadmap，无法直接参考 |

## 四、结论

**借鉴度 ★★ 中**。它是"边缘 agent 运行时"的新锐项目，对我们的价值在**工程方法论**：
① contract-first（单一 schema 源 → 多端生成 + CI 防漂移）；
② 图工作流引擎（节点/边/状态机组织数据管线）；
③ 离线优先 + 硬件/MQTT 一等公民（理念印证）。
LLM 本体不借鉴。建议作为数据管线组织与类型契约设计的参考。
