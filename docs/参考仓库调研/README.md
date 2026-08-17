# 参考仓库调研

> **调研日期**：2026-08-17
> **调研来源**：GitHub 搜索「边缘计算」出现的仓库
> **调研目的**：评估外部项目有哪些点值得借鉴/不值得借鉴，结合本项目
> （三相电流边缘采集器 · current-edge-monitor）的具体模块与待办落地。
>
> 每个仓库对应一份独立总体报告（含可借鉴点+理由、不可借鉴点+理由）。

## 调研策略与进度（广撒网 → 筛选）

> **策略**：广撒网，先中文后英文，最终筛选出约 **中文 10 套 + 英文 10 套** 仓库，
> 提炼可落地到本算法的点。

| 批次 | 目标 | 状态 |
|------|------|------|
| 中文 | ~10 套 | ✅ **已调研 22 套，达成目标（超额 12 套）**；有效 **18 套**；淘汰 4 套（Edge-Intelligence / Wqiankun / star-edge-cloud / EtherCAT） |
| 英文 | ~10 套 | 🔄 **已启动，调研 5 套，有效 4/10**（StreamPipes / Azure IIoT / edge-agents / **mcp-server-mcsa**）；rishvanjay/MCSA 已淘汰 |

> **筛选规则**：主表只保留 ★★ 及以上（可直接借鉴）；★ 低 / ✗ 移入『不推荐关注』（编号保留原调研序，空号=已淘汰，不占有效计数）。总调研 **27 套**，有效 **22 套**，淘汰 **5 套**。

## 汇总

| # | 仓库 | 技术栈 | 一句话定位 | 借鉴等级 |
|---|------|--------|-----------|---------|
| 1 | [freeioe/freeioe](2026-08-17_freeioe.md) | Lua + skynet | 生产级工业 IoT 边缘网关运行时 | ★★★ 高 |
| 2 | [zhangkaigod2000/BetterIOT](2026-08-17_betteriot.md) | C#/.NET Core | ARM/X86 工业数据采集系统 | ★★★ 高 |
| 3 | [iweidujiang/java-industrial-smart](2026-08-17_java-industrial-smart.md) | Java/Spring Boot | 工业智能专栏（PLC接入/预判/监控/孪生） | ★★★ 高 |
| 4 | [dingdaoyi/LotusBridge](2026-08-17_lotusbridge.md) | Rust + axum | 边缘网关（学习项目） | ★★ 中（架构思路） |
| 5 | ~~Edge-Intelligence~~（已淘汰） | | | |
| 6 | ~~Wqiankun/-~~（已淘汰） | | | |
| 7 | [anviod/edgeCore](2026-08-17_edgecore.md) | Go + Vue3 | 工业边缘网关（RK3588 + 稳定性工程） | ★★★ 高（**最高契合**） |
| 8 | [data-infra/cube-studio](2026-08-17_cube-studio.md) | K8s + Python | 一站式 AI/MLOps 云平台 | ★★ 中（远期 2 点） |
| 9 | ~~star-edge-cloud~~（已淘汰） | | | |
| 10 | [zhangedwin/aiotec](2026-08-17_aiotec.md) | C++ + lighttpd | RTU 采集 + 视觉 AI 融合网关（RK3588） | ★★ 中（理念印证） |
| 11 | ~~EtherCAT~~（已淘汰） | | | |
| 12 | [qianyu-web/edge-demo](2026-08-17_edge-demo.md) | Python + EMQX + Node-RED | 工业边缘计算 Demo（PLC采集+异常检测+断网补传） | ★★★ 高（对标 M1） |
| 13 | [huxinyu190/EMS-Modbus-Gateway-](2026-08-17_ems-modbus-gateway.md) | Python + MySQL | 储能 Modbus 采集网关（多类型解码+SOH 健康度） | ★★★ 高（遥测参照） |
| 14 | [Newdawn01/esp32_icm42607](2026-08-17_esp32_icm42607.md) | ESP32-S3 + Rust 网关 | 振动监测边缘节点（EMA 自适应阈值+四层温补） | ★★★ 高（阈值工程） |
| 15 | [chgttyyr/MoonSpectrum](2026-08-17_moonspectrum.md) | MoonBit | 科学信号处理库（FFT/窗/滤波/PSD/STFT） | ★★★ 高（算法清单核对） |
| 16 | [ExpressGit/EdgeAI-Engine](2026-08-17_edgeai-engine.md) | Python/PyTorch + RKNN | RK3588 视觉训练+量化源码（RKNN 路径） | ★★ 中（M3 备查） |
| 17 | [David-gby/PTCG-](2026-08-17_ptcg.md) | Python + YOLOv8 | 卡牌质检（手动修正→训练池→再训练闭环） | ★★ 中（数据飞轮） |
| 18 | [13115827885/PFLD 疲劳驾驶](2026-08-17_pfld-fatigue-driving.md) | Python/PyTorch/TFLite | 疲劳检测（滑窗时序融合降误报+边缘部署管线） | ★★ 中（降误报方法论） |
| 19 | [apache/streampipes](2026-08-17_streampipes.md) | Java + TS | Apache 工业 IoT 流处理平台（管道元素+变化检测） | ★★★ 高（流处理标准件） |
| 20 | [pnoker/iot-dc3](2026-08-17_iot-dc3.md) | Java/Spring Cloud | 工业 IoT 平台（28 协议驱动+Driver SDK+AI） | ★★★ 高（驱动+物模型） |
| 21 | [dgiot/iotStudio](2026-08-17_iotstudio.md) | Python/FastAPI + Vue3 | Python 轻量边缘平台（本体引擎+存储降级） | ★★★ 高（Python 栈边缘） |
| 22 | [IoTSharp/SonnetDB](2026-08-17_sonnetdb.md) | C#/.NET 嵌入式 | 多模型数据引擎（时序+KV+MQ 一进程） | ★★★ 高（嵌入式时序存储） |
| 23 | [Azure/Industrial-IoT](2026-08-17_azure-industrial-iot.md) | C# + OPC UA | 微软 OPC Publisher 边缘模块 | ★★ 中（OPC UA 接入） |
| 24 | [ForestHubAI/edge-agents](2026-08-17_edge-agents.md) | Go + TS | 30MB 边缘 AI agent 运行时 | ★★ 中（contract-first） |
| 25 | [KuzinHouse/IIoT-Edge-Gateway](2026-08-17_iiot-edge-gateway.md) | TS/Next.js | Neuron 级网关（设备模板含变频器+标签告警） | ★★ 中（遥测+告警配置） |
| 26 | [LGDiMaggio/mcp-server-mcsa](2026-08-17_mcp-server-mcsa.md) | Python + MCP | MCSA 完整工具链（故障频率公式+严重度分级+包络） | ★★★ 高（MCSA 核心参考） |

## 总体结论（4 条可直接落地）

1. **特征日志常存 / 数据飞轮的存储架构** → 借鉴 freeioe（COV 变化检测 + 周期/文件缓冲 + 离线冲刷）
   与 BetterIOT（LiteDB 本地缓存 + Sended 清理 + MQTT 断线重连）。对应 TODO B·M1 剩余项。
2. **数据预判流程骨架**（历史阈值 → 实时预判 → 批量落库 → 周期重算基线）→ 借鉴
   java-industrial-smart 08-data-prediction。与已定方向「基线 + 置信区间 + 累积度量」同源，
   可直接映射为特征日志 + 健康度落库。
3. **采集驱动抽象 + 物模型**（统一遥测接入 + 特征规范）→ 借鉴 BetterIOT `IDrive` /
   LotusBridge `Protocol` trait + 物模型设计。对应 TODO「按工况打标签」与 P2 特征规范。
4. **运行时架构（单一事实源 + 调度内核 + 稳定性工程）** → 借鉴 **edgeCore**（最高契合）：
   ShadowCore 内存影子 SoT（特征/状态统一内存层）、ScanEngine 快慢路径调度器、
   DataPipeline 扇出 + 限速背压、熔断/失败降级 + Soak 指标门禁。
   对应 M1 收尾与鲁棒性 R10/R11/R13 及验收门禁。
5. **异常判定 + 断网补传最小可对照实现** → 借鉴 **edge-demo**：滑动窗口统计 + 阈值/3σ 双重判定、
   SQLite 本地缓存 + 断网按序补传（零丢失）+ 7 天 TTL、遥测通道配置。对标 M1 工程实现。
6. **VFD/PLC 遥测接入 + 健康度** → 借鉴 **EMS-Modbus-Gateway-**：多类型寄存器解码表（U16/S16/U32/S32/FLOAT32/BOOL）、
   点位地址映射、SOH 健康状态计算、1/10 分钟分级入库、离线包搬运。
7. **自适应阈值 + 温漂补偿** → 借鉴 **esp32_icm42607**：EMA 基线追踪 + 温度感知自适应阈值、
   四层温度漂移补偿（上电校准→在线学习→温补→基线感知）、3σ 剔除、报警延迟确认。
   对应 TriggerEngine 基线策略与鲁棒性温漂处理。
8. **算法能力清单交叉核对** → 借鉴 **MoonSpectrum** 的纯算法层模块划分（FFT/STFT/PSD/IIR/FIR/窗/卷积/相关/重采样/峰值检测），
   逐一核对 01_信号预处理 + 02_特征 的覆盖完整性。
9. **降误报时序融合** → 借鉴 **PFLD 疲劳驾驶**：滑窗 + 连续帧确认 + 累积率（PERCLOS）三层时序融合、
   绿/黄/红多级告警状态机。对应 TriggerEngine 确认去抖与 03/04 决策设计。
10. **流处理管道元素 + 变化检测处理器** → 借鉴 **Apache StreamPipes**：一个功能一个可独立部署单元
    （adapter/processor/sink，可跑在边缘）；现成处理器库（change-detection / statistics / aggregation /
    pattern-detection）直接对照我们 02/03 模块。
11. **采集驱动抽象 + 物模型的工业化实现** → 借鉴 **iot-dc3**（28 协议驱动 + Driver SDK 注册模式）与
    **iotStudio**（Python 栈边缘代理 + Site→Gateway→Device→Point 四层物模型 + MQTT 主题规范 +
    SQLite 默认/TDengine 可选降级）。对应「按工况打标签」与 P2 特征规范（结论 3 的权威佐证）。
12. **嵌入式时序存储 + 批量写入** → 借鉴 **SonnetDB**：嵌入式时序 + 内建 MQTT 直连落库、
    Line Protocol/JSON 批量写入、WAL 崩溃安全分级。对应特征日志/断网续传的存储选型升级方向。
13. **OPC UA 接入（M3/遥测备查）** → 借鉴 **Azure Industrial-IoT**（OPC Publisher：PubSub 多编码 +
    发布/控制面分离）；若 VFD 走 OPC UA 则深入读其 OPC Publisher 文档。
14. **契约单一来源 + 图工作流** → 借鉴 **edge-agents**：OpenAPI schema 单一事实源 + Go/TS 双端生成 +
    CI 防 drift；图工作流（节点/边/状态机）组织数据管线。
15. **VFD 遥测模板 + 标签告警三元组** → 借鉴 **IIoT-Edge-Gateway**：设备模板（90+ 含变频器）+ 标签告警
    配置（阈值/死区/延迟）。对应 TriggerEngine 迟滞/确认参数与 VFD 接入。
16. **MCSA 故障理论 + 严重度分级（核心参考）** → 借鉴 **mcp-server-mcsa**：断条 (1±2s)·fs、偏心 fs±k·fr、
    定子 fs±2k·fr、轴承 fs±k·f_defect 完整公式；严重度阈值（dB below fundamental：健康 ≤−50 / 初期 −50~−45 /
    中度 −45~−40 / 严重 >−35，按基线实测调整）；Hilbert 包络解调；数据 ID 引用而非搬运。
    对应 M2 健康度分级与 03 检测模型的理论/实现双参考。

## 不推荐关注（已从主表筛选淘汰，保留报告备查）

- **Wqiankun/-**（RL 卸载 toy）：与电流监测无交集，跳过。
- **Edge-Intelligence**：仅"端云分层推断"概念可提，工程实现粗糙，不借鉴。
- **star-edge-cloud**：早期半成品，仅"职责拆分 + 数据分类"架构思路（已被 freeioe/BetterIOT 覆盖），不深入学习。
- **EtherCAT**：工业总线协议 SDK，与电流监测不在一条线；仅"软件模拟从站测测试 + 性能基准量化"两个方法论可记（已纳入总体结论 4 的验证思路）。
- **[rishvanjay/MCSA](2026-08-17_rishvanjay-mcsa.md)**：MCSA ML 脚本集（研究生实验代码，无文档未维护），方法枚举 + 实验数据备查；工程价值被 mcp-server-mcsa 覆盖。

## 理念印证（行业共识，不深入学习）

- **aiotec**：RTU + 视觉 AI 融合网关。未引入新技术点，但第三次印证三个行业共识：断网续传、多协议统一接入、**误报反馈闭环（数据飞轮理念的现成落地形态）**。
- **cube-studio**：云侧 MLOps 平台，与本项目不同层（详见远期备查）。

## 远期备查（非当前）

- **cube-studio**：云侧 MLOps 平台，与本项目不同层。中期（M2–M3 数据飞轮成型后）复用两点：
  模型注册/版本化 + 标注平台接入（届时参考其设计，不部署其平台）。
