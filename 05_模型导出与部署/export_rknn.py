"""
ONNX → RKNN 模型转换脚本

将 ONNX 模型转换为瑞芯微 RKNN 格式，并在 NPU 上运行推理。

前提条件:
  1. 在 RK3588S 设备上运行（或使用 RKNN-Toolkit2 的模拟器）
  2. 已安装 rknn-toolkit2: pip install rknn-toolkit2

转换流程:
  ONNX → 预编译/量化 → RKNN → 部署到 RK3588S NPU

模型精度选择:
  - FP16: 精度几乎无损，速度较慢
  - INT8:  需要校准数据集，速度和功耗最优（推荐）
  - INT4:  精度损失较大，仅适合极端低功耗场景

用法:
    python export_rknn.py --onnx models/arc_cnn.onnx --output models/arc_cnn.rknn --quantize int8
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List, Tuple

# 添加项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# RKNN 转换函数
# ============================================================

def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    dataset_path: Optional[str] = None,
    quantized_dtype: str = "int8",
    target_platform: str = "rk3588",
    optimization_level: int = 3,
) -> str:
    """
    将 ONNX 模型转换为 RKNN 格式

    Args:
        onnx_path: ONNX 模型路径
        output_path: 输出 RKNN 文件路径 (.rknn)
        dataset_path: INT8 量化校准数据集路径（包含 .npy 文件）
        quantized_dtype: 量化类型 "fp16" | "int8" | "int4"
        target_platform: 目标平台 "rk3588" | "rk356x" 等
        optimization_level: 优化等级 0-3

    Returns:
        RKNN 文件路径
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError(
            "需要安装 rknn-toolkit2:\n"
            "  从 https://github.com/airockchip/rknn-toolkit2 下载安装"
        )

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rknn = RKNN(verbose=False)

    # ── 步骤 1: 配置 ──
    print("[1/5] Configuring RKNN...")
    rknn.config(
        mean_values=[[0, 0, 0]],  # 归一化均值（如已在预处理中完成，则设0）
        std_values=[[1, 1, 1]],   # 归一化标准差
        target_platform=target_platform,
        optimization_level=optimization_level,
        quantized_dtype=quantized_dtype,
        # 量化算法
        quantized_algorithm="normal",  # normal / mmse / kl_divergence
        # 混合量化（部分层 FP16，部分 INT8，平衡精度和速度）
        quantized_method="channel",    # layer / channel
    )

    # ── 步骤 2: 加载 ONNX ──
    print("[2/5] Loading ONNX model...")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx 失败, ret={ret}")

    # ── 步骤 3: 构建 RKNN ──
    print("[3/5] Building RKNN model...")

    if quantized_dtype == "int8" and dataset_path:
        # INT8 量化需要校准数据集
        print(f"     Using calibration dataset from: {dataset_path}")
        ret = rknn.build(
            do_quantization=True,
            dataset=dataset_path,  # 校准数据 .txt 文件路径（每行一个 .npy 路径）
        )
    else:
        ret = rknn.build(do_quantization=(quantized_dtype != "fp16"))

    if ret != 0:
        raise RuntimeError(f"RKNN build 失败, ret={ret}")

    # ── 步骤 4: 导出 RKNN ──
    print("[4/5] Exporting RKNN model...")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export 失败, ret={ret}")

    # ── 步骤 5: 精度分析 (可选) ──
    print("[5/5] Analyzing accuracy...")
    try:
        ret = rknn.accuracy_analysis(
            inputs=[os.path.join(dataset_path, f) for f in os.listdir(dataset_path)[:5]]
            if dataset_path else None
        )
        if ret == 0:
            print("     Accuracy analysis completed.")
    except Exception as e:
        print(f"     [WARN] Accuracy analysis skipped: {e}")

    rknn.release()

    print(f"\n[OK] RKNN model exported to: {output_path}")
    print(f"     Quantization:     {quantized_dtype}")
    print(f"     Target platform:  {target_platform}")
    print(f"     File size:        {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def prepare_calibration_dataset(
    numpy_files_dir: str,
    num_samples: int = 100,
) -> str:
    """
    准备 INT8 量化校准数据集

    从包含 .npy 文件的目录中生成校准数据集列表文件。

    Args:
        numpy_files_dir: 包含预处理后波形 .npy 文件的目录
        num_samples: 用于校准的样本数（通常 100-500 即可）

    Returns:
        校准数据集文件路径
    """
    npy_files = [f for f in os.listdir(numpy_files_dir) if f.endswith(".npy")]

    if len(npy_files) == 0:
        raise ValueError(f"目录中没有 .npy 文件: {numpy_files_dir}")

    # 随机采样
    if len(npy_files) > num_samples:
        npy_files = np.random.choice(npy_files, num_samples, replace=False).tolist()

    # 写入列表文件
    dataset_file = os.path.join(numpy_files_dir, "calibration_dataset.txt")
    with open(dataset_file, "w") as f:
        for npy_file in npy_files:
            f.write(os.path.join(numpy_files_dir, npy_file) + "\n")

    print(f"[INFO] Calibration dataset prepared: {len(npy_files)} samples → {dataset_file}")
    return dataset_file


# ============================================================
# RKNN 模型信息查看
# ============================================================

def inspect_rknn_model(rknn_path: str):
    """
    查看 RKNN 模型信息

    Args:
        rknn_path: RKNN 模型文件路径
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        raise ImportError("需要安装 rknn-toolkit2")

    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(rknn_path)

    if ret != 0:
        print(f"[ERROR] 无法加载 RKNN 模型: {rknn_path}")
        return

    # 获取模型信息
    try:
        # SDK 版本
        sdk_version = rknn.get_sdk_version()
        print(f"SDK Version: {sdk_version}")

        # 模型输入输出信息通过其他方式获取
    except Exception as e:
        print(f"[WARN] 部分信息获取失败: {e}")

    rknn.release()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ONNX → RKNN 模型转换")
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="ONNX 模型路径"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出 RKNN 文件路径"
    )
    parser.add_argument(
        "--quantize", type=str, default="int8",
        choices=["fp16", "int8", "int4"],
        help="量化类型 (推荐: int8)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="INT8 校准数据集目录路径"
    )
    parser.add_argument(
        "--platform", type=str, default="rk3588",
        choices=["rk3588", "rk356x", "rk3562", "rk3576"],
        help="目标平台"
    )
    parser.add_argument(
        "--opt_level", type=int, default=3,
        choices=[0, 1, 2, 3],
        help="优化等级"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="仅查看模型信息，不转换"
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_rknn_model(args.onnx)
        return

    convert_onnx_to_rknn(
        onnx_path=args.onnx,
        output_path=args.output,
        dataset_path=args.dataset,
        quantized_dtype=args.quantize,
        target_platform=args.platform,
        optimization_level=args.opt_level,
    )


if __name__ == "__main__":
    main()
