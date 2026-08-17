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
| 中文 | ~10 套 | ✅ 已调研 9 套；高价值筛出 **4 套**（freeioe / BetterIOT / java-industrial-smart / edgeCore），再补 ~1 套达成 |
| 英文 | ~10 套 | ⏳ 待开始 |

## 汇总

| # | 仓库 | 技术栈 | 一句话定位 | 借鉴等级 |
|---|------|--------|-----------|---------|
| 1 | [freeioe/freeioe](2026-08-17_freeioe.md) | Lua + skynet | 生产级工业 IoT 边缘网关运行时 | ★★★ 高 |
| 2 | [zhangkaigod2000/BetterIOT](2026-08-17_betteriot.md) | C#/.NET Core | ARM/X86 工业数据采集系统 | ★★★ 高 |
| 3 | [iweidujiang/java-industrial-smart](2026-08-17_java-industrial-smart.md) | Java/Spring Boot | 工业智能专栏（PLC接入/预判/监控/孪生） | ★★★ 高 |
| 4 | [dingdaoyi/LotusBridge](2026-08-17_lotusbridge.md) | Rust + axum | 边缘网关（学习项目） | ★★ 中（架构思路） |
| 5 | [wyc941012/Edge-Intelligence](2026-08-17_edge-intelligence.md) | Python/PyTorch | 端云 CNN 分层推断（学术 demo） | ★ 低（概念） |
| 6 | [Wqiankun/-](2026-08-17_wqiankun-rl-offloading.md) | Python/gym | RL 计算卸载（学术 toy） | ✗ 不相关 |
| 7 | [anviod/edgeCore](2026-08-17_edgecore.md) | Go + Vue3 | 工业边缘网关（RK3588 + 稳定性工程） | ★★★ 高（**最高契合**） |
| 8 | [data-infra/cube-studio](2026-08-17_cube-studio.md) | K8s + Python | 一站式 AI/MLOps 云平台 | ★★ 中（远期 2 点） |
| 9 | [mythad/star-edge-cloud](2026-08-17_star-edge-cloud.md) | Go | 边缘-云监测平台（早期半成品） | ★ 低（架构思路） |

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

## 不推荐关注

- **Wqiankun/-**（RL 卸载 toy）：与电流监测无交集，跳过。
- **Edge-Intelligence**：仅"端云分层推断"概念可提，工程实现粗糙，不借鉴。
- **star-edge-cloud**：早期半成品，仅"职责拆分 + 数据分类"架构思路（已被 freeioe/BetterIOT 覆盖），不深入学习。

## 远期备查（非当前）

- **cube-studio**：云侧 MLOps 平台，与本项目不同层。中期（M2–M3 数据飞轮成型后）复用两点：
  模型注册/版本化 + 标注平台接入（届时参考其设计，不部署其平台）。
