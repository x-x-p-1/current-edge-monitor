# Azure/Industrial-IoT — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/Azure/Industrial-IoT
> **技术栈**：C#/.NET + OPC UA PubSub + Azure IoT Edge + Docker + MIT（577★ / 221 fork）
> **定位**：微软 OPC Publisher 边缘模块（OPC UA 资产发现/注册/遥测发布 + Azure 云接入）
> **契合度**：★★ 中（OPC UA 接入与发布/控制面分离；依赖 Azure 生态，仅方法论）

## 一、仓库概览

Microsoft Industrial IoT 平台，核心是 **OPC Publisher**——运行于本地（on-premises）的 Azure IoT Edge 模块，用于发现、注册并管理 OPC UA 资产。OPC Publisher 是**完全合规的 OPC UA PubSub 遥测发布器**（支持 JSON、JSON+Gzip、UADP 二进制编码），并通过控制面提供大部分 OPC UA 服务。

核心特性（README 核实）：
- OPC UA PubSub 遥测发布（JSON / JSON+Gzip / UADP 二进制编码）
- 控制面：HTTP(s)（Preview）、MQTT Broker（Preview）、Azure IoT Hub device methods
- Azure IoT Edge 模块化部署（on-premises）
- 预构建 Docker 容器（MCR）
- 严格版本化发布策略（semver；仅支持最新 patch；安全更新打到最后 patch 版本）
- 59 贡献者、46 个 release，支持策略文档化

## 二、值得借鉴（为什么）

### 1. OPC UA PubSub 接入与编码（M3/遥测备查）
- 借鉴：OPC UA PubSub 发布器，多编码（JSON/Gzip/UADP）。
- 为什么：若 VFD/PLC 走 OPC UA 遥测，其 PubSub 发布/订阅、编码与订阅管理是标准参照（OPC UA 是工业数据接入的事实标准）。

### 2. 发布面与控制面分离
- 借鉴：遥测发布（数据面）与控制接口（HTTP/MQTT/Device Methods）分离。
- 为什么：印证"数据面 + 控制面分离"架构——我们采集器的数据上报与配置/控制通道可分离设计。

### 3. 版本化发布与支持策略
- 借鉴：semver + 仅支持最新 patch + 安全更新策略文档化。
- 为什么：工程治理参照（我们 4 个 commit 起步，未来版本化策略可参考）。

### 4. 容器化边缘部署
- 借鉴：IoT Edge 模块化 + 预构建镜像。
- 为什么：边缘模块化部署的工业实践（我们 RK3588 虽原生部署为主，容器化是可选路径）。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| Azure IoT Hub 云接入 | 我们无 Azure 依赖 |
| C#/.NET 技术栈 | 我们 Python 栈 |
| Azure 部署脚本/云服务 | 平台绑定 |
| IoT Edge 运行时 | 我们目标是原生/轻量部署 |

## 四、结论

**借鉴度 ★★ 中**。它是 OPC UA 工业接入的官方级实现，价值在**OPC UA PubSub 编码/发布架构**（M3/遥测接入备查）与**发布/控制面分离**思想。
依赖 Azure 生态不采用，仅方法论。若我们 VFD 走 OPC UA，深入读其 OPC Publisher 文档；否则作为遥测协议备查。
