"""
RKNN 推理运行时模块

在 RK3588S 边缘设备上使用 RKNN-Toolkit2-Lite 进行 NPU 推理。

与 PC 端 rknn-toolkit2 不同，rknn-toolkit2-lite 是轻量版本，
不包含模型转换功能，只提供推理 API。

安装:
    pip install rknn-toolkit2-lite

推理流程:
    1. 加载 .rknn 模型文件
    2. 预处理输入波形 → numpy 数组 [1, 1, 256]
    3. 调用 NPU 推理
    4. 解析输出结果

性能参考 (RK3588S, INT8):
    - 1D-CNN (5层):     ~0.3ms/帧
    - 1D-CNN (8层):     ~0.5ms/帧
    - AutoEncoder:      ~0.4ms/帧
    - 1D-ResNet (6层):  ~0.6ms/帧
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# NPU 推理器
# ============================================================

class NPUInferenceError(Exception):
    """NPU 推理错误"""
    pass


class RKNNInferenceEngine:
    """
    RKNN NPU 推理引擎

    用法:
        engine = RKNNInferenceEngine("models/arc_cnn_int8.rknn")
        input_data = np.random.randn(1, 1, 256).astype(np.float32)
        output = engine.infer(input_data)

    Args:
        model_path: .rknn 模型文件路径
        npu_core_mask: NPU 核心掩码 (0=auto, 1=core0, 2=core1, 3=core2)
    """

    def __init__(self, model_path: str, npu_core_mask: int = 0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN 模型文件不存在: {model_path}")

        self.model_path = model_path
        self.npu_core_mask = npu_core_mask
        self._rknn = None
        self._input_info = None
        self._output_info = None
        self._warmup_done = False

        self._init_rknn()

    def _init_rknn(self):
        """初始化 RKNN Runtime"""
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "需要安装 rknn-toolkit2-lite (边缘端推理库):\n"
                "  pip install rknn-toolkit2-lite"
            )

        self._rknn = RKNNLite(verbose=False)

        # 加载模型
        ret = self._rknn.load_rknn(self.model_path)
        if ret != 0:
            raise NPUInferenceError(f"加载 RKNN 模型失败, ret={ret}")

        # 初始化 Runtime（NPU 核心）
        ret = self._rknn.init_runtime(
            core_mask=self._get_core_mask(),
            perf_debug=False,
        )
        if ret != 0:
            raise NPUInferenceError(f"初始化 RKNN Runtime 失败, ret={ret}")

        print(f"[INFO] RKNN Engine initialized: {self.model_path}")

    def _get_core_mask(self) -> int:
        """
        获取 NPU 核心掩码

        RK3588S NPU 有 3 个核心:
          core_mask = 0b001 → core0
          core_mask = 0b010 → core1
          core_mask = 0b100 → core2
          core_mask = 0b111 → 自动分配 (推荐)
        """
        if self.npu_core_mask == 0:
            return 0b111  # 自动
        elif self.npu_core_mask == 1:
            return 0b001
        elif self.npu_core_mask == 2:
            return 0b010
        elif self.npu_core_mask == 3:
            return 0b100
        else:
            return 0b111

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        执行 NPU 推理

        Args:
            input_data: 输入数据，shape 需匹配模型的输入定义
                       通常为 [batch, 1, length] float32

        Returns:
            输出列表，包含所有输出节点的结果
        """
        if self._rknn is None:
            raise NPUInferenceError("RKNN Runtime 未初始化")

        # 确保数据类型和形状正确
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # 推理
        start_time = time.perf_counter()
        outputs = self._rknn.inference(inputs=[input_data])
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000  # 微秒

        # 第一次推理作为 warmup（包括 kernel 编译等开销）
        if not self._warmup_done:
            self._warmup_done = True
            # Warmup 后重新测量
            outputs = self._rknn.inference(inputs=[input_data])
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000

        # 记录延迟（纳秒级精度）
        self._last_inference_time_us = elapsed_us

        return outputs

    def infer_batch(self, input_batch: np.ndarray) -> List[np.ndarray]:
        """
        批量推理（如果模型支持 batch > 1）

        Args:
            input_batch: [batch_size, 1, length] float32

        Returns:
            批量输出
        """
        return self.infer(input_batch)

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 1, 256),
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        NPU 推理性能基准测试

        Args:
            input_shape: 输入形状
            num_runs: 测试运行次数

        Returns:
            性能统计字典
        """
        times_us = []

        for i in range(num_runs):
            dummy = np.random.randn(*input_shape).astype(np.float32)
            _ = self.infer(dummy)
            times_us.append(self._last_inference_time_us)

        times_us = np.array(times_us)

        # 去除前 10% 作为 warmup
        warmup_cut = num_runs // 10
        if warmup_cut > 0:
            times_us = times_us[warmup_cut:]

        stats = {
            "mean_us": float(np.mean(times_us)),
            "median_us": float(np.median(times_us)),
            "min_us": float(np.min(times_us)),
            "max_us": float(np.max(times_us)),
            "std_us": float(np.std(times_us)),
            "p99_us": float(np.percentile(times_us, 99)),
            "num_runs": len(times_us),
        }

        return stats

    @property
    def last_inference_time_us(self) -> float:
        """上一次推理的耗时（微秒）"""
        return getattr(self, "_last_inference_time_us", 0.0)

    def close(self):
        """释放 NPU 资源"""
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __del__(self):
        self.close()


# ============================================================
# 多模型管理器
# ============================================================

class MultiModelManager:
    """
    多模型管理器 — 同时管理多个 RKNN 模型

    在边缘设备上一键加载所有检测模型，通过名称索引进行推理。
    """

    def __init__(self, model_configs: Dict[str, str]):
        """
        Args:
            model_configs: {模型名称: .rknn 文件路径} 字典
                e.g. {
                    "arc_cnn": "models/arc_cnn_int8.rknn",
                    "anomaly_ae": "models/anomaly_ae_int8.rknn",
                    "load_cls": "models/load_classifier_int8.rknn",
                }
        """
        self._engines: Dict[str, RKNNInferenceEngine] = {}

        for name, path in model_configs.items():
            if os.path.exists(path):
                try:
                    self._engines[name] = RKNNInferenceEngine(path)
                    print(f"[INFO] Loaded model '{name}': {path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load model '{name}': {e}")
            else:
                print(f"[WARN] Model file not found, skipped '{name}': {path}")

    def infer(self, model_name: str, input_data: np.ndarray) -> List[np.ndarray]:
        """使用指定模型进行推理"""
        if model_name not in self._engines:
            raise KeyError(f"模型 '{model_name}' 未加载")
        return self._engines[model_name].infer(input_data)

    def infer_all(
        self, input_data: np.ndarray
    ) -> Dict[str, List[np.ndarray]]:
        """使用所有已加载模型进行推理"""
        results = {}
        for name, engine in self._engines.items():
            results[name] = engine.infer(input_data)
        return results

    def benchmark_all(
        self, input_shape: Tuple[int, ...] = (1, 1, 256)
    ) -> Dict[str, Dict[str, float]]:
        """所有模型性能基准测试"""
        results = {}
        for name, engine in self._engines.items():
            results[name] = engine.benchmark(input_shape)
        return results

    def close_all(self):
        """释放所有模型"""
        for engine in self._engines.values():
            engine.close()
        self._engines.clear()

    @property
    def loaded_models(self) -> List[str]:
        return list(self._engines.keys())

    def __del__(self):
        self.close_all()
