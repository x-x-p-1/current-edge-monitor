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
    engine = AcquisitionEngine(pp, wd, trig, cap)
    for chunk in adc_stream:
        saved = engine.feed(chunk, t_s=t, meta={"state": "LOAD"})
        for info in saved:
            print("切片已保存:", info["path"])
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


class AcquisitionEngine:
    """feed 驱动的 M1 采集主循环。"""

    def __init__(
        self,
        preprocessor,
        watchdog,
        trigger,
        capture,
        ring_capacity: Optional[int] = None,
    ):
        """
        Args:
            preprocessor: CurrentPreprocessor（流式预处理）
            watchdog: WatchdogFeatures
            trigger: TriggerEngine
            capture: SliceCapture
            ring_capacity: 环形缓冲容量（点数）。缺省 = max(pre+post, 32768)，
                保证预触发在补录后触发期间不被覆盖。
        """
        from .ring_buffer import RingBuffer

        self.pp = preprocessor
        self.wd = watchdog
        self.trig = trigger
        self.cap = capture
        self.channels = int(getattr(self.pp, "channels", 3))
        if ring_capacity is None:
            ring_capacity = max(
                capture.ctx.pre_samples + capture.ctx.post_samples,
                32768,
            )
        self.buffer = RingBuffer(ring_capacity, self.channels)
        self._pending: Optional[Dict] = None

    def reset(self) -> None:
        """重置全部状态（设备重启 / 重新锚定）"""
        self.buffer.reset()
        self.wd.reset()
        self.trig.reset()
        self._pending = None

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
        # 1) 原始采样全速率入环（常开）
        self.buffer.write(raw_chunk)

        # 2) 流式预处理 → 帧
        frames = self.pp.process_streaming(raw_chunk)

        saved: List[Dict] = []
        for frame in frames:
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
        return saved
