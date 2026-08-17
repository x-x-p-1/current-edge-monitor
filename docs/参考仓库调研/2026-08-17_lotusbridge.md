# dingdaoyi/LotusBridge — 参考报告

> **调研日期**：2026-08-17
> **仓库**：https://github.com/dingdaoyi/LotusBridge
> **技术栈**：Rust + axum + sqlx(SQLite) + tokio + Flutter(桌面) + MQTT
> **定位**：Rust 边缘计算设备网关（学习项目，功能较浅）

## 一、仓库概览

LotusBridge 是一个 Rust 学习项目型边缘网关，核心思路清晰但完成度有限（只做通 modbus）：
- **协议抽象**：`Protocol` trait + `ProtocolStore` 注册表，feature 编译期扩展（modbus-tcp / modbus-rtu）
- **数据模型**：Device / Point / DeviceGroup / DataType，物模型化的点位组织
- **导出抽象**：`DataExport` trait（南向采集 → 北向推送），已实现"消智云"MQTT 导出
- **存储**：SQLite + axum REST API + 鉴权；另附 Flutter 桌面管理端
- **设计文档**：`docs/边缘网关需求分析.md` 专门讨论了"边缘网关是否需要物模型"

## 二、值得借鉴（为什么）

### 1. 物模型设计（Device → Point → 属性/单位/数据类型）
对应 P2 特征规范（命名/维度/数值域/可解释性）。
- 借鉴：把"设备-点位-属性-单位-数据类型"组织成统一物模型，值带单位/时间戳。
- 为什么：我们特征日志目前是"看门狗快照 + 切片"，字段命名未标准化；用物模型组织
  （每特征 = 一个 point，带单位/维度/quality）能让特征日志、切片 meta、ML 特征向量
  三处共用一套规范，正好支撑 P2 特征规范落地。

### 2. `Protocol` trait + `ProtocolStore` 注册表
与 BetterIOT/Java 的驱动抽象互相印证（第三种语言实现同一模式）。
- 借鉴：协议注册表 + trait 接口，新增协议只加实现不改造核心。
- 为什么：多份独立仓库不约而同采用"接口 + 注册表"扩展模式，说明是行业共识。

### 3. 南向/北向解耦（DataExport trait）
对应数据飞轮"采集与导出解耦"。
- 借鉴：采集（南向）与推送（北向）各走一个 trait，可独立替换。
- 为什么：我们将来"特征日志 → 本地/云上送"可插拔，不必绑死一种输出。

### 4. SQLite 轻量落库
对应特征日志本地存储候选。
- 借鉴：边缘单机用 SQLite（无服务、单文件）存点位/配置/最近数据。
- 为什么：比 LiteDB（.NET）/Redis（Java）更轻，Python 的 sqlite3 零依赖，是特征日志
  落库的最轻选择之一。

## 三、不需要借鉴（为什么）

| 点 | 为什么不需要 |
|----|-------------|
| Rust + axum 工程（学习级） | 完成度低（仅 modbus、规则引擎未做），生产参考价值不如 BetterIOT/Java；我们也不换技术栈 |
| feature 编译期插件扩展 | 单板单算法，不需要编译期动态插件 |
| GPL-3.0 许可 | 若直接抄代码要注意许可证传染；仅借鉴设计理念无碍 |
| Flutter 桌面管理端 | 当前阶段无桌面管理需求，产品化再说 |

## 四、结论

**借鉴度 ★★ 中**。最大价值是**物模型设计**（对 P2 特征规范有直接帮助），
其次印证了"接口 + 注册表"驱动抽象是行业共识。其余架构被 BetterIOT 覆盖，完成度也不足。
