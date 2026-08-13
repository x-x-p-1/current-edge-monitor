"""
09 — 采集层（M1 主线，P0）
==========================
面向边缘端实时运行的采集/捕获模块（纯软件，可仿真验证）：

  - ring_buffer.py   全速率环形缓冲（固定容量、常开、O(1) 写入）
  - watchdog.py      毫秒级看门狗特征（RMS / 峰值包络 / 峰值因子 / RMS 斜率）
  - trigger.py       触发引擎（stopping time：EWMA 基线 + K·σ 统计判据 + 去抖确认）
  - slice_capture.py 切片捕获（预触发 + 后触发落盘，携带上下文 → 数据飞轮）
  - engine.py        AcquisitionEngine：feed 驱动的 M1 运行时主循环

与 01 预处理的关系：v2.1 的流式状态延续滤波（慢基线 + 带通）保证
快路径帧保留基波，看门狗/触发依赖绝对幅值，必须使用幅值保持的帧
（norm_enabled=False）。
"""
import importlib

_ring = importlib.import_module("09_采集层.ring_buffer")
RingBuffer = _ring.RingBuffer

_watch = importlib.import_module("09_采集层.watchdog")
WatchdogFeatures = _watch.WatchdogFeatures
WatchdogSnapshot = _watch.WatchdogSnapshot

_trigger = importlib.import_module("09_采集层.trigger")
TriggerEngine = _trigger.TriggerEngine
TriggerEvent = _trigger.TriggerEvent

_slice = importlib.import_module("09_采集层.slice_capture")
SliceCapture = _slice.SliceCapture
CaptureContext = _slice.CaptureContext

_engine = importlib.import_module("09_采集层.engine")
AcquisitionEngine = _engine.AcquisitionEngine

__all__ = [
    "RingBuffer",
    "WatchdogFeatures",
    "WatchdogSnapshot",
    "TriggerEngine",
    "TriggerEvent",
    "SliceCapture",
    "CaptureContext",
    "AcquisitionEngine",
]
