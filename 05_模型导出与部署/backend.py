"""
推理后端抽象 (v2 新增)
======================
防"板卡锁定"的部署层抽象：同一套算法/模型可在不同推理后端上运行，
切换后端 = 改一个配置项（runtime.backend），而非重写部署层。

后端：
  - onnx_cpu   : onnxruntime，通用便携（PC 开发 / A733 无 NPU 场景）
  - rknn       : RKNNLite，RK3588S NPU
  - allwinner  : Allwinner NPU（占位，A733 接入时实现）

设计依据：项目"两阶段板卡策略"——Phase 1 用 RK3588 跑通，Phase 2 用 A733 量产；
模型一律以 ONNX 为交换格式，后端可替换（详见《板卡选型与两阶段策略》讨论）。
"""

import numpy as np
from typing import List, Optional


class InferenceError(Exception):
    """推理后端错误"""
    pass


class InferenceBackend:
    """推理后端抽象接口"""

    name: str = "base"

    def load(self) -> None:
        """加载模型。"""
        raise NotImplementedError

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """执行推理。返回输出张量列表。"""
        raise NotImplementedError


class OnnxCpuBackend(InferenceBackend):
    """onnxruntime CPU 后端（通用便携，开发 / 无 NPU 场景）"""

    name = "onnx_cpu"

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise InferenceError("需要安装 onnxruntime: pip install onnxruntime")
        self._session = ort.InferenceSession(self.model_path, providers=self.providers)

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        if self._session is None:
            self.load()
        x = np.asarray(input_data, dtype=np.float32)
        inp = {self._session.get_inputs()[0].name: x}
        return [np.asarray(o) for o in self._session.run(None, inp)]


class RknnBackend(InferenceBackend):
    """RKNN NPU 后端（RK3588S）"""

    name = "rknn"

    def __init__(self, model_path: str, npu_core_mask: int = 0):
        self.model_path = model_path
        self.npu_core_mask = npu_core_mask
        self._rknn = None

    def load(self) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise InferenceError("需要安装 rknn-toolkit2-lite（边缘端推理库）")
        self._rknn = RKNNLite(verbose=False)
        if self._rknn.load_rknn(self.model_path) != 0:
            raise InferenceError(f"加载 RKNN 模型失败: {self.model_path}")
        core_mask = 0b111 if self.npu_core_mask == 0 else self.npu_core_mask
        if self._rknn.init_runtime(core_mask=core_mask) != 0:
            raise InferenceError("初始化 RKNN Runtime 失败")

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        if self._rknn is None:
            self.load()
        x = np.asarray(input_data, dtype=np.float32)
        out = self._rknn.inference(inputs=[x])
        return [np.asarray(o) for o in out]


class AllwinnerBackend(InferenceBackend):
    """Allwinner NPU 后端（A733 占位，接入时实现）"""

    name = "allwinner"

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs

    def load(self) -> None:
        raise InferenceError("Allwinner 后端尚未实现（A733 接入时开发）")

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        raise InferenceError("Allwinner 后端尚未实现")


def create_inference_backend(name: str, model_path: str, **kwargs) -> InferenceBackend:
    """推理后端工厂（由配置 runtime.backend 驱动）。

    Args:
        name: "onnx_cpu" | "rknn" | "allwinner"
        model_path: 模型文件路径
        **kwargs: 后端特定参数（如 npu_core_mask）

    Returns:
        InferenceBackend 实例
    """
    name = name.lower()
    if name == "onnx_cpu":
        return OnnxCpuBackend(model_path, **kwargs)
    if name == "rknn":
        return RknnBackend(model_path, **kwargs)
    if name == "allwinner":
        return AllwinnerBackend(model_path, **kwargs)
    raise ValueError(f"未知推理后端: {name}（可选 onnx_cpu / rknn / allwinner）")
