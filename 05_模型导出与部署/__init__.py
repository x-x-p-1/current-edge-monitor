"""模型部署模块"""
from .export_onnx import export_to_onnx, verify_onnx
from .export_rknn import convert_onnx_to_rknn, prepare_calibration_dataset
from .rknn_inference import RKNNInferenceEngine, MultiModelManager, NPUInferenceError
from .backend import (
    InferenceBackend,
    InferenceError,
    OnnxCpuBackend,
    RknnBackend,
    AllwinnerBackend,
    create_inference_backend,
)

__all__ = [
    "export_to_onnx",
    "verify_onnx",
    "convert_onnx_to_rknn",
    "prepare_calibration_dataset",
    "RKNNInferenceEngine",
    "MultiModelManager",
    "NPUInferenceError",
    # 推理后端抽象 (v2)
    "InferenceBackend",
    "InferenceError",
    "OnnxCpuBackend",
    "RknnBackend",
    "AllwinnerBackend",
    "create_inference_backend",
]
