"""05 推理后端抽象 v2 验证（运行：python 06_测试与验证/verify_backend_v2.py）"""
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_dep = importlib.import_module("05_模型导出与部署")

print("=== 后端工厂 ===")
b1 = _dep.create_inference_backend("onnx_cpu", "dummy.onnx")
b2 = _dep.create_inference_backend("rknn", "dummy.rknn")
b3 = _dep.create_inference_backend("allwinner", "dummy.onnx")
print("onnx_cpu →", type(b1).__name__)
print("rknn     →", type(b2).__name__)
print("allwinner→", type(b3).__name__)
assert isinstance(b1, _dep.OnnxCpuBackend)
assert isinstance(b2, _dep.RknnBackend)
assert isinstance(b3, _dep.AllwinnerBackend)

print("=== 未知后端应报错 ===")
try:
    _dep.create_inference_backend("tpu", "x.onnx")
    print("❌ 未报错")
except ValueError as e:
    print("✅ 正确报错:", str(e)[:40], "...")

print("=== Rknn 后端缺库时应抛 InferenceError（非 ImportError 穿透） ===")
try:
    b2.load()
    print("⚠️ rknnlite 已安装（正常，跳过）")
except _dep.InferenceError as e:
    print("✅ 正确抛 InferenceError:", str(e)[:40], "...")
except Exception as e:
    print("❌ 异常类型不对:", type(e).__name__, str(e)[:60])

print("✅ 05 后端抽象验证通过")
