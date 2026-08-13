"""
10 — 鲁棒性（运行时加固）
==========================
对齐《08_鲁棒性/鲁棒性清单.md》的软件实现：

  - guards.py         数值守卫（R1 NaN/Inf、R2 除零）
  - input_quality.py  输入质量自评（R3 削波 / R4 偏置·断线 / R5 丢样 / R6 缺相·错序）
  - chaos.py          混沌注入器（§4.2 fault injection）

与 09 采集层的集成：AcquisitionEngine 内嵌数值守卫 + 输入质量门控 + 单帧隔离（R10）。
"""
import importlib

_guards = importlib.import_module("10_鲁棒性.guards")
frame_is_valid = _guards.frame_is_valid
sanitize_nan_inf = _guards.sanitize_nan_inf
safe_divide = _guards.safe_divide
safe_log10 = _guards.safe_log10
NumericGuard = _guards.NumericGuard

_iq = importlib.import_module("10_鲁棒性.input_quality")
InputQuality = _iq.InputQuality
QualityFlags = _iq.QualityFlags

_chaos = importlib.import_module("10_鲁棒性.chaos")
ChaosInjector = _chaos.ChaosInjector

__all__ = [
    "frame_is_valid", "sanitize_nan_inf", "safe_divide", "safe_log10",
    "NumericGuard",
    "InputQuality", "QualityFlags",
    "ChaosInjector",
]
