"""
切片捕获（M1 采集层 — 预触发 + 后触发落盘）
=============================================
触发引擎判定异常（stopping time）后，从全速率环形缓冲中冻结
**预触发**波形，并继续采集 **后触发**波形，形成完整切片落盘，
携带上下文（时戳 + 触发原因 + 状态 + VFD 遥测）→ 未来 ML 训练数据
（数据飞轮：触发事件切片 = 训练数据）。

落盘格式: np.savez_compressed 单文件：
  - data:      (N, C) 波形（预触发 + 后触发，时间顺序）
  - 全部上下文以命名数组保存（fs / pre_samples / reason / meta ...）

用法（配合 AcquisitionEngine / 手动）:
    cap = SliceCapture(CaptureContext(pre_samples=4096, post_samples=8192,
                                      out_dir="slices"))
    # 触发后继续写 post_samples 点，然后：
    info = cap.capture(buffer, event, meta={"state": "LOAD", ...})
    # → 返回保存路径 + 切片信息，文件已落盘
"""
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class CaptureContext:
    """切片捕获配置"""

    pre_samples: int = 4096        # 预触发样本数（来自环形缓冲）
    post_samples: int = 8192       # 后触发样本数（触发后继续采集）
    sample_rate: float = 16000.0
    channels: int = 3
    out_dir: str = "slices"        # 切片输出目录（相对运行目录或绝对路径）


class SliceCapture:
    """从环形缓冲捕获切片并落盘。"""

    def __init__(self, ctx: CaptureContext):
        self.ctx = ctx
        os.makedirs(ctx.out_dir, exist_ok=True)
        self._seq = 0

    def capture(
        self,
        buffer,
        event,
        meta: Optional[Dict] = None,
        trigger_abs_index: Optional[int] = None,
    ) -> Dict:
        """冻结切片并落盘。

        Args:
            buffer: RingBuffer（预触发波形来源）
            event: TriggerEvent（触发原因 / 时戳）
            meta: 附加上下文（状态机状态、VFD 遥测、温度等）
            trigger_abs_index: 触发点的绝对索引。缺省用 buffer 当前写入点
                （即调用时最新样本处）。

        Returns:
            info: dict {path, n_samples, pre_avail, post_samples, reason, ts}
                  pre_avail < pre_samples 表示预触发被缓冲容量裁剪（不完整）。
        """
        if trigger_abs_index is None:
            trigger_abs_index = buffer._count  # 触发点 = 当前最新样本之后

        pre = self.ctx.pre_samples
        post = self.ctx.post_samples
        start = trigger_abs_index - pre
        stop = trigger_abs_index + post
        data = buffer.slice_range(start, stop)

        pre_avail = max(0, trigger_abs_index - buffer.first_available_index())
        pre_avail = min(pre_avail, pre)

        ts = getattr(event, "timestamp_s", time.time())
        reason = getattr(event, "reason", "unknown")

        # 相对时间戳（如仿真 0..N 秒）→ 用墙钟，便于归档排序
        if float(ts) < 1e9:
            ts = time.time()

        self._seq += 1
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(float(ts)))
        fname = f"slice_{self._seq:05d}_{reason}_{ts_str}.npz"
        path = os.path.join(self.ctx.out_dir, fname)

        np.savez_compressed(
            path,
            data=data.astype(np.float32),
            sample_rate=np.float64(self.ctx.sample_rate),
            channels=np.int32(self.ctx.channels),
            pre_samples=np.int32(pre),
            pre_avail=np.int32(pre_avail),
            post_samples=np.int32(post),
            trigger_abs_index=np.int64(trigger_abs_index),
            reason=np.array(reason),
            timestamp_s=np.float64(ts),
            meta=str(meta or {}),
        )
        return {
            "path": path,
            "n_samples": int(data.shape[0]),
            "pre_avail": int(pre_avail),
            "post_samples": int(post),
            "reason": reason,
            "ts": float(ts),
        }
