# 06 — 测试与验证模块手册

## 模块概述

包含算法栈各层级的单元测试、集成测试和端到端测试，确保每一层功能正确、边界安全。

```
单元测试 → 集成测试 → 端到端测试 → 性能基准
(每层独立)  (层间联调)  (完整管线)  (时序约束)
```

---

## 测试文件清单

| 文件 | 测试范围 | 用例数 | 依赖 |
|------|---------|--------|------|
| `test_preprocess.py` | 信号预处理 | 9 | numpy, scipy |
| `test_features.py` | 特征提取 | 18 | numpy, scipy |
| `test_models.py` | AI 检测模型 | 8 | torch |
| `test_postprocess.py` | 后处理决策 | 14 | numpy |
| `test_integration.py` | 端到端集成（v2） | 4 | numpy, scipy |

**总计：57 个测试用例**（v2 集成：模拟→预处理→快/慢特征→状态机→迟滞/事件）

---

## 相关工具

各层 v2 的功能验证脚本、M0 仿真、实时看板已移至 **`07_测试工具/`**（见该模块 README），
本目录只保留正式回归套件。

## 运行测试

```bash
# 运行全部测试
pytest 06_测试与验证/ -v

# 按模块运行
pytest 06_测试与验证/test_preprocess.py -v    # 预处理
pytest 06_测试与验证/test_features.py -v      # 特征提取
pytest 06_测试与验证/test_models.py -v        # AI 模型
pytest 06_测试与验证/test_postprocess.py -v   # 后处理
pytest 06_测试与验证/test_integration.py -v   # 端到端

# 快速模式（仅非 AI 测试）
pytest 06_测试与验证/test_preprocess.py 06_测试与验证/test_features.py 06_测试与验证/test_postprocess.py -v
```

---

## 测试详情

### 1. 信号预处理测试 (test_preprocess.py)

| 测试类 | 测试用例 | 验证内容 |
|--------|---------|----------|
| `TestFilters` | `test_remove_dc_offset` | DC 去除后均值 ≈ 0 |
| | `test_bandpass_filter` | 高频分量被衰减 |
| | `test_savitzky_golay_smooth` | 输出长度不变 |
| `TestNormalization` | `test_zscore_normalization` | μ≈0, σ≈1 |
| | `test_minmax_normalization` | min=0, max=1 |
| | `test_params_consistency` | 参数保存/应用结果一致 |
| `TestAlignment` | `test_find_zero_crossings` | 能检测到过零点 |
| | `test_align_to_zero_crossing` | 对齐后起点≈0 |
| `TestPreprocessPipeline` | `test_full_pipeline` | 端到端预处理管线正常 |

### 2. 特征提取测试 (test_features.py)

| 测试类 | 测试用例 | 验证内容 |
|--------|---------|----------|
| `TestTimeDomain` | `test_rms_sine` | 正弦波 RMS ≈ 0.707 |
| | `test_peak_factor_sine` | 正弦波 CF ≈ 1.414 |
| | `test_form_factor_sine` | 正弦波 FF ≈ 1.111 |
| | `test_kurtosis_normal` | 正态分布峭度 ≈ 0 |
| | `test_zero_crossing_rate` | 过零率 > 0 |
| | `test_differential_stats` | 差分统计含所有字段 |
| | `test_extract_all` | 完整特征提取正常 |
| `TestFrequencyDomain` | `test_spectrum` | 频谱维度正确 |
| | `test_harmonics` | 基波有幅值 |
| | `test_thd` | THD > 0 |
| | `test_high_freq_energy_ratio` | 正弦波 HF ≈ 0 |
| | `test_band_energies` | 5 个频段能量 |
| | `test_spectral_statistics` | 质心 > 0 |
| `TestStatistical` | `test_quartiles` | q25 < q50 < q75 |
| | `test_boxplot_summary` | 含所有五数概括 |
| | `test_boxplot_skewness` | 正态数据偏度 ≈ 0 |
| | `test_outlier_detection` | 能检测到离群点 |
| | `test_trend_slope` | 递增趋势斜率 > 0 |

### 3. 检测模型测试 (test_models.py)

| 测试类 | 测试用例 | 验证内容 |
|--------|---------|----------|
| `TestArcDetectionCNN` | `test_forward_shape` | 输出 [batch, 2] |
| | `test_predict` | 概率和=1 |
| | `test_model_factory` | 工厂函数正确创建 |
| | `test_rule_based_detection` | 规则基线正常 |
| `TestAutoEncoder` | `test_encode_decode_shape` | 编码-解码形状匹配 |
| | `test_reconstruction_error` | 重构误差 > 0 |
| `TestLoadClassifier` | `test_forward_shape` | 输出 [batch, 8] |
| | `test_predict` | 概率正确 |
| `TestPowerQuality` | `test_frequency_estimation` | 频率 ≈ 50Hz |
| | `test_half_cycle_rms` | 半周期 RMS 正常 |
| | `test_power_quality_report` | 报告格式正确 |

### 4. 后处理测试 (test_postprocess.py)

| 测试类 | 测试用例 | 验证内容 |
|--------|---------|----------|
| `TestHysteresisAlarm` | `test_normal_no_trigger` | 正常值不报警 |
| | `test_single_spike_no_alarm` | 单次尖峰不报警 |
| | `test_continuous_trigger` | 连续≥3次 → ALARM |
| | `test_release_after_alarm` | 连续低值 → NORMAL |
| | `test_no_release_during_alarm` | 偶发低值不解警 |
| `TestMultiLevelHysteresis` | 3 个用例 | 多级状态转移 |
| `TestEventAggregator` | 2 个用例 | 事件合并/分离 |
| `TestDecisionFusion` | 2 个用例 | 融合正确性 |
| `TestScoreSmoother` | 2 个用例 | 平滑正确性 |

### 5. 集成测试 (test_integration.py)

| 测试用例 | 验证内容 |
|---------|----------|
| `test_end_to_end_normal` | 正常波形全管线不误报 |
| `test_end_to_end_arc` | 电弧波形全管线正确检出 |
| `test_pipeline_with_alarm` | 100 帧模拟管线 + 报警事件 |
| `test_timing_constraint` | 总延迟 < 5ms |

---

## 已知限制

### 短窗口统计偏差

256 点 @ 50kSPS 仅覆盖 0.256 个工频周期。这导致：
- FFT 频率分辨率仅 195Hz，频谱泄露严重
- RMS/峰值因子等统计量有小幅偏差

**解决方案**：
- 单元测试使用 2000 点（2 个完整周期）验证算法正确性
- 实际应用窗口（256 点）的偏差已通过容差设置适配

### PyTorch/numpy 版本兼容

当前环境 numpy 2.5.1 与 torch 2.13.0 不兼容。运行模型测试前需降级 numpy：
```bash
pip install "numpy<2"
```

---

## 添加新测试

```python
# 模板：在对应测试文件中添加
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # 准备测试数据
        pass
    
    def test_basic_functionality(self):
        """基本功能验证"""
        result = your_function(input_data)
        self.assertIsNotNone(result)
    
    def test_edge_cases(self):
        """边界条件"""
        # 空输入
        # 极值输入
        # 单点输入
        pass
    
    def test_performance(self):
        """性能约束"""
        import time
        start = time.perf_counter()
        for _ in range(1000):
            your_function(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertLess(elapsed_ms, 100)  # < 100ms
```

---

## 测试覆盖矩阵

| 功能 | 单位 | 集成 | 边界 | 性能 |
|------|------|------|------|------|
| DC 去除 | ✅ | ✅ | ✅ | — |
| 带通滤波 | ✅ | ✅ | — | — |
| 归一化 | ✅ | ✅ | ✅ | — |
| 时域特征 | ✅ | ✅ | ✅ | — |
| 频域特征 | ✅ | ✅ | ✅ | — |
| 电弧 CNN | ✅ | ⏳ | — | ⏳ |
| AutoEncoder | ✅ | ⏳ | — | ⏳ |
| 负载识别 | ✅ | ⏳ | — | — |
| 电能质量 | ✅ | ⏳ | — | — |
| 迟滞报警 | ✅ | ✅ | ✅ | — |
| 事件聚合 | ✅ | ✅ | ✅ | — |
| 决策融合 | ✅ | ✅ | ✅ | — |
