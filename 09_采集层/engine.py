"""
采集引擎（M1 主线 — 运行时主循环雏形）
========================================
把 01 预处理流式管线 + 环形缓冲 + 看门狗特征 + 触发引擎 + 切片捕获
串成 feed 驱动的实时主循环：

    feed(raw_chunk)
      ├─ 原始采样全速率写入环形缓冲（常开）
      ├─ 流式预处理 → 帧（状态延续滤波，v2.1）
      ├─ 看门狗特征（每帧）
      ├─ 触发判定（stopping time + 去抖）
      └─ 触发后：继续采集 post_samples → 冻结切片落盘（携带上下文）

用法:
    engine = AcquisitionEngine(pp, wd, trig, cap, input_quality=iq)
    for chunk in adc_stream:
        saved = engine.feed(chunk, t_s=t, meta={"state": "LOAD"})
        for info in saved:
            print("切片已保存:", info["path"])

鲁棒性（10_鲁棒性）:
  - R1  NaN/Inf 清洗后入环（不污染环形缓冲/滤波状态）
  - R3-6 输入质量门控：劣化帧标记并跳过诊断（不误报）
  - R7  质量/异常日志有界（deque，无内存增长）
  - R10 单帧异常隔离：任何一帧异常不杀进程，记录并跳过
"""
import importlib
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

_rob = importlib.import_module("10_鲁棒性")
sanitize_nan_inf = _rob.sanitize_nan_inf
InputQuality = _rob.InputQuality


class AcquisitionEngine:
    """feed 驱动的 M1 采集主循环。"""

    def __init__(
        self,
        preprocessor,
        watchdog,
        trigger,
        capture,
        ring_capacity: Optional[int] = None,
        input_quality: Optional[InputQuality] = None,
        quality_log_size: int = 64,
    ):
        """
        Args:
            preprocessor: CurrentPreprocessor（流式预处理）
            watchdog: WatchdogFeatures
            trigger: TriggerEngine
            capture: SliceCapture
            ring_capacity: 环形缓冲容量（点数）。缺省 = max(pre+post, 32768)，
                保证预触发在补录后触发期间不被覆盖。
            input_quality: InputQuality 实例（None = 关闭输入质量门控）
            quality_log_size: 质量/异常日志容量（R7 内存有界）
        """
        from .ring_buffer import RingBuffer

        self.pp = preprocessor
        self.wd = watchdog
        self.trig = trigger
        self.cap = capture
        self.fs = float(getattr(self.pp, "sample_rate", 16000.0))
        self.channels = int(getattr(self.pp, "channels", 3))
        if ring_capacity is None:
            ring_capacity = max(
                capture.ctx.pre_samples + capture.ctx.post_samples,
                32768,
            )
        self.buffer = RingBuffer(ring_capacity, self.channels)
        self.quality: Optional[InputQuality] = input_quality
        self._pending: Optional[Dict] = None
        self._quality_log: Deque[Tuple[str, Optional[float], object]] = deque(
            maxlen=quality_log_size)
        self._errors = 0
        self._degraded_frames = 0
        self._nan_sanitized = 0

    def reset(self) -> None:
        """重置全部状态（设备重启 / 重新锚定）"""
        self.buffer.reset()
        self.wd.reset()
        self.trig.reset()
        self._pending = None
        self._errors = 0
        self._degraded_frames = 0
        self._nan_sanitized = 0
        if self.quality is not None:
            self.quality.reset()
        self._quality_log.clear()

    @property
    def quality_log(self) -> list:
        """质量/异常日志（有界）"""
        return list(self._quality_log)

    def feed(
        self,
        raw_chunk: np.ndarray,
        t_s: Optional[float] = None,
        meta: Optional[Dict] = None,
    ) -> List[Dict]:
        """喂入一段原始采样，驱动整条 M1 链路。

        Args:
            raw_chunk: 原始采样，(M,) 或 (M, C)
            t_s: 当前时间（秒）
            meta: 附加上下文（过程状态、VFD 遥测等），触发落盘时携带

        Returns:
            本次喂入期间新保存的切片信息列表（无则 []）
        """
        # R10: 非数组/标量输入（突发异常/断流）→ 记录并跳过，不杀进程
        if raw_chunk is None or np.asarray(raw_chunk).ndim == 0:
            self._errors += 1
            self._quality_log.append(("non_array_input", t_s, ""))
            return []
        # 空块（丢样/无新采样）→ 安全跳过
        if np.asarray(raw_chunk).size == 0:
            self._quality_log.append(("empty_chunk", t_s, ""))
            return []

        # R1: NaN/Inf 清洗后入环（不污染缓冲与滤波状态）
        chunk, has_nan = sanitize_nan_inf(raw_chunk)
        if has_nan:
            self._nan_sanitized += 1
            self._quality_log.append(("nan_sanitized", t_s, ""))

        # R3-6: 输入层质量自评（原始采样，块级 —— 断线/偏置/削波在原始域即可检出，
        # 与滤波状态无关；避免"从正常切到常值"时慢基线瞬态掩盖断线）
        skip_diagnosis = False
        if self.quality is not None:
            q = self.quality.evaluate(chunk, self.fs, t_s)
            if not q.valid:
                # 硬无效（断线/NaN）：保存原始，跳过本块诊断（降级，不误报）
                self._degraded_frames += 1
                self._quality_log.append((q.summary, t_s, q.score))
                skip_diagnosis = True
            elif q.score < 0.8:
                # 软劣化（削波/偏置/丢样/缺相等）：标记但继续（鲁棒性清单：标记+继续）
                self._quality_log.append(("degraded:" + q.summary, t_s, q.score))

        # 原始始终入环（切片捕获/旁路保存仍可用）
        self.buffer.write(chunk)
        if skip_diagnosis:
            return []   # 输入不可用：保存原始，跳过本块诊断（降级，不误报）

        # 2) 流式预处理 → 帧
        frames = self.pp.process_streaming(chunk)

        saved: List[Dict] = []
        for frame in frames:
            try:
                # 3) 看门狗特征
                snap = self.wd.update(frame, t_s=t_s)
                # 4) 触发判定
                ev = self.trig.update(snap, t_s=t_s)

                # 5) 后触发补录 → 切片落盘
                if ev.triggered:
                    self._pending = {
                        "trigger_idx": self.buffer._count,
                        "event": ev,
                        "meta": dict(meta or {}),
                    }
                if self._pending is not None:
                    if (self.buffer._count - self._pending["trigger_idx"]
                            >= self.cap.ctx.post_samples):
                        info = self.cap.capture(
                            self.buffer,
                            self._pending["event"],
                            meta=self._pending["meta"],
                            trigger_abs_index=self._pending["trigger_idx"],
                        )
                        saved.append(info)
                        self._pending = None
            except Exception as exc:  # noqa: BLE001
                # R10: 单帧隔离——任何一帧异常不杀进程，记录并跳过
                self._errors += 1
                self._quality_log.append(("frame_exception", t_s, repr(exc)[:120]))
                continue
        return saved
