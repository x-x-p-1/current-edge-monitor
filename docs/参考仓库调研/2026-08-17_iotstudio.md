# dgiot/iotStudio — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/dgiot/iotStudio
> **技术栈**：Python + FastAPI + Vue3 + SQLite/TDengine + MQTT + Parse-lite（275★ / 60 fork）
> **定位**：DGIOT Lite 物联网边缘应用框架（Python 轻量版边缘代理 + 协议采集 + 本体引擎）
> **契合度**：★★★ 高（Python 栈边缘代理 + 物模型 + 存储降级，最贴近我们架构）

## 一、仓库概览

iotStudio 是 DG-IoT 的 Python 轻量版（DGIOT Lite）：物联网边缘应用框架，Python + FastAPI + Vue3，作为**边缘代理**负责协议采集与解析，可联动 DG-IoT 中枢平台。

架构定位（README 核实）：
```
iotStudio (边缘)        DG-IoT (中枢)          iotStudio (应用)
Python · 轻量代理       Erlang · 高性能         Vue3 · 低代码
协议·采集·解析 ──MQTT──▶ EMQX 汇聚 ──REST──▶ 12页管理后台/7插件
<1000 设备/节点         >10万 设备汇聚          用户交互层
```

核心能力：
- **数据采集**：Modbus TCP/RTU（多从站轮询/寄存器扫描）、A11（油气 5a5a 帧）、OPC UA（订阅+轮询）、OPC DA、IEC 104
- **数据存储**：SQLite 默认（零安装）、PostgreSQL（生产多租户可选）、TDengine 3.x 时序（可选，**无则降级 SQLite**）
- **Parse-lite**：Parse Server Python 兼容层（CRUD/查询/用户/角色/ACL/CLP/批量/Hooks）
- **本体引擎**：四层模型 Site → Gateway → Device → Point，`sync_to_parse()` 自动建 Device 对象，MQTT `dgiot/{site}/{gateway}/{device}/{point}/data`
- **多租户**：X-Tenant-ID header + JWT
- 管理后台 12 页 7 组（仪表盘/设备/组态/数据分析/告警/流计算/预测性维护/报文解析/通道管理/边缘代理/MQTT 调试/模拟器/系统）
- 插件架构：`manifest.js` 按需启用（Vue tree-shaking），支持自定义厂商协议接入

## 二、值得借鉴（为什么）

### 1. Python 栈边缘代理（最贴近我们架构）
- 借鉴：FastAPI 轻量边缘代理 + SQLite 默认 + TDengine 可选降级。
- 为什么：这是中文项目里**最接近我们技术选型**（Python + SQLite + MQTT）的边缘实现——"默认零依赖 SQLite，时序引擎可选降级"的策略可直接采纳。

### 2. 四层本体引擎（Site→Gateway→Device→Point）
- 借鉴：四层物模型 + 自动同步 + MQTT 主题规范。
- 为什么：我们「按工况打标签 + 物模型」可直接采用这套层级（站点→网关→设备→测点）与主题命名，是 BetterIOT/LotusBridge 之外的第三个具体物模型。

### 3. 存储降级策略
- 借鉴：TDengine 可用则用，不可用降级 SQLite。
- 为什么：对应我们"边缘 SQLite 保底 + 可选时序引擎升级"的存储策略。

### 4. 插件按需启用（tree-shaking）
- 借鉴：manifest.js 控制插件加载，协议插件按需。
- 为什么：我们 M1/M2 的协议驱动与模块化可参考"按需启用 + tree-shaking"，减小板端体积。

### 5. 仿真/调试工具内置
- 借鉴：协议模拟器 + MQTT 调试工具。
- 为什么：印证"仿真器打底 + 自检工具"理念（同我们 00 仿真 + 07 自检）。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| Parse Server 兼容层 | 我们不需要移动端云端后端 |
| 多租户 | 单站部署 |
| 与 DG-IoT 中枢联动 | 我们无自有中枢，走通用 MQTT |
| Vue 管理后台本体 | 我们看板是自定义 HTML |

## 四、结论

**借鉴度 ★★★ 高**。Python 栈边缘代理 + 四层物模型 + 存储降级，是**与我们技术选型最贴近的边缘实现**。
直接采纳：① SQLite 默认 + 时序可选降级；② Site→Gateway→Device→Point 物模型与 MQTT 主题；③ 插件按需启用。
建议作为「采集驱动抽象 + 物模型」结论的 Python 侧落地参照。
