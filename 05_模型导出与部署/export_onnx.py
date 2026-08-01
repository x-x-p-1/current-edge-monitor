"""
PyTorch → ONNX 模型导出脚本

将 PyTorch 训练好的模型导出为 ONNX 格式，
作为 RKNN 转换的中间步骤。

支持模型:
  - ArcDetectionCNN (电弧检测)
  - ArcDetectionCNN_LSTM (电弧检测 CNN+LSTM)
  - CurrentAutoEncoder (异常检测)
  - LoadClassifier1DResNet (负载识别)

用法:
    python export_onnx.py --model arc_cnn --checkpoint models/best.pth --output models/arc_cnn.onnx
"""

import os
import sys
import argparse
import importlib

import torch
import torch.nn as nn
from typing import Tuple

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 数字开头目录无法标准导入，用 importlib 运行时加载
_arc_mod = importlib.import_module("03_检测模型.arc_detection")
_anomaly_mod = importlib.import_module("03_检测模型.anomaly_detection")
_load_mod = importlib.import_module("03_检测模型.load_identification")
ArcDetectionCNN = _arc_mod.ArcDetectionCNN
ArcDetectionCNN_LSTM = _arc_mod.ArcDetectionCNN_LSTM
CurrentAutoEncoder = _anomaly_mod.CurrentAutoEncoder
LoadClassifier1DResNet = _load_mod.LoadClassifier1DResNet


# ============================================================
# 导出函数
# ============================================================

def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int, int],
    output_path: str,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    opset_version: int = 12,
) -> str:
    """
    导出 PyTorch 模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        input_shape: 输入张量形状 (batch, channels, length)，如 (1, 1, 256)
        output_path: 输出 ONNX 文件路径
        input_names: 输入节点名称
        output_names: 输出节点名称
        dynamic_axes: 动态轴配置
        opset_version: ONNX opset 版本

    Returns:
        ONNX 文件路径
    """
    model.eval()

    # 创建 dummy input
    dummy_input = torch.randn(*input_shape)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        # 默认: batch 维度为动态
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print(f"[OK] ONNX model exported to: {output_path}")
    print(f"     Input shape:  {input_shape}")
    print(f"     File size:    {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def verify_onnx(onnx_path: str, input_shape: Tuple[int, int, int]) -> bool:
    """
    验证 ONNX 模型

    Args:
        onnx_path: ONNX 文件路径
        input_shape: 输入形状

    Returns:
        是否验证通过
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] 未安装 onnx / onnxruntime，跳过验证。")
        print("       安装: pip install onnx onnxruntime")
        return True

    # 检查模型结构
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model structure verified.")

    # 测试推理
    session = ort.InferenceSession(onnx_path)
    dummy_input = torch.randn(*input_shape).numpy()
    outputs = session.run(None, {"input": dummy_input})
    print(f"[OK] ONNX inference test passed. Output shape: {outputs[0].shape}")

    return True


# ============================================================
# 模型构建与导出
# ============================================================

def export_arc_cnn(
    checkpoint_path: str,
    output_dir: str = "models",
    input_length: int = 256,
):
    """导出电弧检测 CNN 模型"""
    model = ArcDetectionCNN(input_length=input_length)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    output_path = os.path.join(output_dir, "arc_cnn.onnx")
    export_to_onnx(model, (1, 1, input_length), output_path)
    verify_onnx(output_path, (1, 1, input_length))


def export_arc_cnn_lstm(
    checkpoint_path: str,
    output_dir: str = "models",
    input_length: int = 256,
):
    """导出电弧检测 CNN+LSTM 模型"""
    model = ArcDetectionCNN_LSTM(input_length=input_length)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    output_path = os.path.join(output_dir, "arc_cnn_lstm.onnx")
    export_to_onnx(model, (1, 1, input_length), output_path)
    verify_onnx(output_path, (1, 1, input_length))


def export_anomaly_ae(
    checkpoint_path: str,
    output_dir: str = "models",
    input_length: int = 256,
    latent_dim: int = 32,
):
    """导出自编码器模型"""
    model = CurrentAutoEncoder(input_length=input_length, latent_dim=latent_dim)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    output_path = os.path.join(output_dir, "anomaly_ae.onnx")
    export_to_onnx(model, (1, 1, input_length), output_path)
    verify_onnx(output_path, (1, 1, input_length))


def export_load_classifier(
    checkpoint_path: str,
    output_dir: str = "models",
    num_classes: int = 8,
    input_length: int = 256,
):
    """导出负载分类器模型"""
    model = LoadClassifier1DResNet(num_classes=num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")

    output_path = os.path.join(output_dir, "load_classifier.onnx")
    export_to_onnx(model, (1, 1, input_length), output_path)
    verify_onnx(output_path, (1, 1, input_length))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PyTorch → ONNX 模型导出")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["arc_cnn", "arc_cnn_lstm", "anomaly_ae", "load_classifier"],
        help="模型类型"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="PyTorch checkpoint 路径 (.pth)"
    )
    parser.add_argument(
        "--output", type=str, default="models",
        help="输出目录"
    )
    parser.add_argument(
        "--input_length", type=int, default=256,
        help="输入波形长度"
    )

    args = parser.parse_args()

    if args.model == "arc_cnn":
        export_arc_cnn(args.checkpoint, args.output, args.input_length)
    elif args.model == "arc_cnn_lstm":
        export_arc_cnn_lstm(args.checkpoint, args.output, args.input_length)
    elif args.model == "anomaly_ae":
        export_anomaly_ae(args.checkpoint, args.output, args.input_length)
    elif args.model == "load_classifier":
        export_load_classifier(args.checkpoint, args.output, input_length=args.input_length)


if __name__ == "__main__":
    main()
