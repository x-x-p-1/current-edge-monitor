# 参考仓库调研

> **调研日期**：2026-08-17
> **调研来源**：GitHub 搜索「边缘计算」出现的仓库
> **调研目的**：评估外部项目有哪些点值得借鉴/不值得借鉴，结合本项目
> （三相电流边缘采集器 · current-edge-monitor）的具体模块与待办落地。
>
> 每个仓库对应一份独立总体报告（含可借鉴点+理由、不可借鉴点+理由）。

## 汇总

| # | 仓库 | 技术栈 | 一句话定位 | 借鉴等级 |
|---|------|--------|-----------|---------|
| 1 | [freeioe/freeioe](2026-08-17_freeioe.md) | Lua + skynet | 生产级工业 IoT 边缘网关运行时 | ★★★ 高 |
| 2 | [zhangkaigod2000/BetterIOT](2026-08-17_betteriot.md) | C#/.NET Core | ARM/X86 工业数据采集系统 | ★★★ 高 |
| 3 | [iweidujiang/java-industrial-smart](2026-08-17_java-industrial-smart.md) | Java/Spring Boot | 工业智能专栏（PLC接入/预判/监控/孪生） | ★★★ 高 |
| 4 | [dingdaoyi/LotusBridge](2026-08-17_lotusbridge.md) | Rust + axum | 边缘网关（学习项目） | ★★ 中（架构思路） |
| 5 | [wyc941012/Edge-Intelligence](2026-08-17_edge-intelligence.md) | Python/PyTorch | 端云 CNN 分层推断（学术 demo） | ★ 低（概念） |
| 6 | [Wqiankun/-](2026-08-17_wqiankun-rl-offloading.md) | Python/gym | RL 计算卸载（学术 toy） | ✗ 不相关 |

## 总体结论（3 条可直接落地）

1. **特征日志常存 / 数据飞轮的存储架构** → 借鉴 freeioe（COV 变化检测 + 周期/文件缓冲 + 离线冲刷）
   与 BetterIOT（LiteDB 本地缓存 + Sended 清理 + MQTT 断线重连）。对应 TODO B·M1 剩余项。
2. **数据预判流程骨架**（历史阈值 → 实时预判 → 批量落库 → 周期重算基线）→ 借鉴
   java-industrial-smart 08-data-prediction。与已定方向「基线 + 置信区间 + 累积度量」同源，
   可直接映射为特征日志 + 健康度落库。
3. **采集驱动抽象 + 物模型**（统一遥测接入 + 特征规范）→ 借鉴 BetterIOT `IDrive` /
   LotusBridge `Protocol` trait + 物模型设计。对应 TODO「按工况打标签」与 P2 特征规范。

## 不推荐关注

- **Wqiankun/-**（RL 卸载 toy）：与电流监测无交集，跳过。
- **Edge-Intelligence**：仅"端云分层推断"概念可提，工程实现粗糙，不借鉴。
