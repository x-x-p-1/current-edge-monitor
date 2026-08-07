"""
三相电流原始数据仿真器 (v0.2)
=============================
用途：在真实采集硬件就绪之前，按《采集侧确认清单》的原始数据契约生成合成
      三相电流原始数据，用于：
        1) 测试预处理/特征/触发/切片全管线（管线自检）
        2) 作为真实采集数据的对照体检基准（SNR/同步性/连续性）
        3) 生成带标签的合成故障样本，为 ML 路线（正常建模/异常检测/分类）打底

v0.2 新增：
  - 物理模式：Motor(默认 5kW) 驱动，幅值单位为真实安培(A)
    load_profile 第三项 = 负载转矩(标幺)，堵转/负载突变/三相不平衡按物理量解释

参考：
  - 西门子 LGF 波形发生算法族（Sinus/SawTooth/Rectangle）的边缘管线自检理念
  - 三菱 MCSA 边带模型（转子条故障边带位于 f1 ± 2·s·f1）

依赖：仅 numpy

用法示例：
    from current_simulator import generate_dataset, Fault, Motor

    # 标幺模式（默认，向后兼容）
    sig, ts, meta = generate_dataset(
        fs=16000, duration=10.0, f1=50.0,
        faults=[Fault(kind="rotor_sideband", start=5.0, dur=3.0, slip=0.03, depth=0.02)],
    )

    # 5kW 电机物理模式（真实安培）：空载→满载→堵转
    sig, ts, meta = generate_dataset(
        duration=10.0, motor=Motor(),
        load_profile=[(0.0, 2.0, 0.0), (2.0, 7.0, 1.0), (7.0, 10.0, 1.0)],
        faults=[Fault(kind="stall", start=4.0, dur=1.0)],
    )
    np.save("sim/abc_normal.npy", sig)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 故障 / 异常定义
# ============================================================

@dataclass
class Fault:
    """合成故障/异常注入定义。

    kind:
      stall          堵转：电流幅值突增（×depth）
      load_step      负载突变：幅值阶跃（×depth）
      rotor_sideband MCSA 转子条边带：注入 f1 ± 2·k·s·f1（k=1..k 阶，强度 depth）
      spike          随机冲击脉冲（接触不良/毛刺），density 控制密度
      harmonic       谐波畸变升高：注入高次谐波（强度 depth）
      unbalance      三相不平衡：B 相降 C 相升
      degradation    全程缓变劣化：幅值线性升至 1+depth（慢性故障）
      arc            电弧：零休（平肩）+ 高频随机爆发（强度 depth）
    """
    kind: str = "stall"
    start: float = 0.0          # 起始时间 (s)
    dur: float = 1.0            # 持续时间 (s)；degradation 全程生效
    depth: float = 0.5          # 幅值倍率 / 强度
    slip: float = 0.03          # (rotor_sideband) 转差率
    k: int = 2                  # (rotor_sideband) 边带阶次上限
    params: dict = field(default_factory=dict)


# ============================================================
# 电机参数（物理模型，默认 5kW）
# ============================================================

@dataclass
class Motor:
    """三相异步电机物理参数（默认 5kW / 380V / 50Hz / 4 极，Y 系列典型值）。

    把仿真从"标幺"升级为"真实安培"：负载转矩(标幺) → 相电流有效值(A)。
    供 generate_dataset(motor=...) 的物理模式使用。

    派生量：
      rated_current_a  : 额定相电流  I = P / (√3·U·η·cosφ)
      no_load_current_a: 空载电流 ≈ no_load_current_pu × 额定
      stall_current_a  : 堵转(锁转子)电流 ≈ stall_current_pu × 额定
      phase_current_a  : 负载转矩(标幺) → 相电流有效值 (A)
    """
    rated_power_w: float = 5000.0      # 额定功率 (W)
    rated_voltage_v: float = 380.0     # 额定线电压 (V)
    frequency_hz: float = 50.0         # 额定频率 (Hz)
    poles: int = 4                     # 极数（4 极 → 同步转速 1500 rpm）
    efficiency: float = 0.87           # 额定效率
    power_factor: float = 0.83         # 额定功率因数
    no_load_current_pu: float = 0.33   # 空载电流 ≈ 33% 额定（Y 系典型）
    stall_current_pu: float = 6.0      # 堵转/起动电流 ≈ 6× 额定（锁转子 ≈ 起动）

    @property
    def rated_current_a(self) -> float:
        """额定相电流 (A)：I = P / (√3·U·η·cosφ)。"""
        return self.rated_power_w / (
            np.sqrt(3.0) * self.rated_voltage_v * self.efficiency * self.power_factor
        )

    @property
    def no_load_current_a(self) -> float:
        return self.rated_current_a * self.no_load_current_pu

    @property
    def stall_current_a(self) -> float:
        return self.rated_current_a * self.stall_current_pu

    def phase_current_a(self, torque_pu: float) -> float:
        """负载转矩(标幺 0~1) → 相电流有效值 (A)。

        模型：磁化电流(≈空载电流)近似恒定 + 转矩分量随负载线性增长
              I(T) = √(I_mag² + (I_rated² - I_mag²)·T²)
        """
        i_mag = self.no_load_current_a
        i_tq = np.sqrt(max(self.rated_current_a ** 2 - i_mag ** 2, 0.0))
        return float(np.sqrt(i_mag ** 2 + (i_tq * float(torque_pu)) ** 2))

    def peak_current_a(self, torque_pu: float) -> float:
        """负载转矩 → 相电流峰值 (√2 × 有效值)。"""
        return np.sqrt(2.0) * self.phase_current_a(torque_pu)

    @property
    def rated_voltage_phase_v(self) -> float:
        """额定相电压 (V)：线电压 / √3（380V → ≈220V）。"""
        return self.rated_voltage_v / np.sqrt(3.0)

    @property
    def rated_voltage_peak_v(self) -> float:
        """额定相电压峰值 (V)。"""
        return np.sqrt(2.0) * self.rated_voltage_phase_v

    def load_power_factor(self, torque_pu: float) -> float:
        """负载转矩(标幺) → 功率因数 cosφ（空载低≈0.20，满载≈额定 0.83）。

        物理：空载时以励磁无功为主(PF 低)，随负载上升转矩分量增大 PF 升高。
        （命名避免与字段 self.power_factor=额定 cosφ 冲突）
        """
        pf_no_load = 0.20
        t = float(np.clip(torque_pu, 0.0, 1.0))
        return float(pf_no_load + (self.power_factor - pf_no_load) * t)


# ============================================================
# 内部工具
# ============================================================

def _make_t(fs: float, duration: float) -> np.ndarray:
    return np.arange(int(duration * fs)) / fs


def _quantize(
    x: np.ndarray,
    adc_bits: int,
    full_scale: float,
    return_clip: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """模拟 ADC 量化与削波。"""
    levels = float(2 ** (adc_bits - 1))
    q = np.clip(x, -full_scale, full_scale)
    q = np.round(q / (full_scale / levels)) * (full_scale / levels)
    if return_clip:
        clipped = np.abs(x) > full_scale
        return q, clipped
    return q, None


def _rescale_to_rms(signal: np.ndarray, seg: slice, target_rms: float) -> None:
    """把信号段(三相)整体缩放到目标相电流有效值 (A)。"""
    seg_rms = float(np.sqrt(np.mean(signal[seg] ** 2)))
    if seg_rms > 1e-9:
        signal[seg] *= (target_rms / seg_rms)


def _apply_fault(
    signal: np.ndarray,
    t: np.ndarray,
    f: Fault,
    fs: float,
    f1: float,
    amplitude: float,
    rng: np.random.Generator,
    motor: Optional[Motor] = None,
) -> np.ndarray:
    n = len(t)
    if f.kind == "degradation":
        ramp = np.linspace(1.0, 1.0 + f.depth, n)
        return signal * ramp[:, None]

    i0 = max(0, int(f.start * fs))
    i1 = min(n, int((f.start + f.dur) * fs))
    if i1 <= i0:
        return signal
    seg = slice(i0, i1)

    # 物理模式（motor 给定）：堵转 / 负载突变 / 三相不平衡 按真实电流解释
    if motor is not None:
        if f.kind == "stall":
            # 堵转：电流跳到堵转电流（≈ 6× 额定）
            _rescale_to_rms(signal, seg, motor.stall_current_a)
            return signal
        if f.kind == "load_step":
            # 负载突变：depth 解释为负载转矩(标幺)，跳到对应相电流
            _rescale_to_rms(signal, seg, motor.phase_current_a(f.depth))
            return signal
        if f.kind == "unbalance":
            # 三相不平衡：depth 解释为电流不平衡度(分数)，B 相降 C 相升
            signal[seg, 1] *= (1.0 - f.depth)
            signal[seg, 2] *= (1.0 + f.depth)
            return signal

    if f.kind in ("stall", "load_step"):
        signal[seg] *= f.depth

    elif f.kind == "rotor_sideband":
        for ch in range(3):
            for kk in range(1, f.k + 1):
                for sign in (+1, -1):
                    fb = f1 + sign * 2 * kk * f.slip * f1
                    signal[seg, ch] += (
                        f.depth * amplitude * np.sin(2 * np.pi * fb * t[seg] + ch)
                    )

    elif f.kind == "spike":
        density = f.params.get("density", 0.001)
        nsp = max(1, int((f.dur * fs) * density))
        idx = rng.integers(i0, i1, nsp)
        for ch in range(3):
            signal[idx, ch] += f.depth * amplitude * rng.choice([-1.0, 1.0], nsp)

    elif f.kind == "harmonic":
        for ch in range(3):
            for kk in (5, 7, 11, 13, 17, 19):
                signal[seg, ch] += (
                    f.depth * amplitude * np.sin(2 * np.pi * kk * f1 * t[seg] + ch)
                )

    elif f.kind == "unbalance":
        signal[seg, 1] *= (1.0 - f.depth)
        signal[seg, 2] *= (1.0 + f.depth)

    elif f.kind == "arc":
        for ch in range(3):
            seg_vals = signal[seg, ch]
            # 零休：过零附近钳位到 0（模拟电弧平肩）
            zero_mask = np.abs(seg_vals) < 0.02 * amplitude
            seg_vals[zero_mask] = 0.0
            # 高频随机爆发（强度随幅值衰减）
            burst = rng.normal(0.0, f.depth * amplitude, seg_vals.shape)
            burst[np.abs(seg_vals) > 0.05 * amplitude] *= 0.1
            seg_vals += burst
            signal[seg, ch] = seg_vals

    else:
        raise ValueError(f"未知故障类型: {f.kind}")

    return signal


# ============================================================
# 主生成接口
# ============================================================

def generate_dataset(
    fs: float = 16000.0,
    duration: float = 10.0,
    f1: float = 50.0,
    amplitude: float = 1.0,
    harmonics: Optional[Dict[int, float]] = None,
    fsw: float = 8000.0,
    pwm_depth: float = 0.03,
    snr_db: float = 60.0,
    adc_bits: int = 24,
    full_scale: Optional[float] = None,
    imbalance: float = 0.0,
    load_profile: Optional[List[Tuple[float, float, float]]] = None,
    faults: Optional[List[Fault]] = None,
    seed: int = 42,
    motor: Optional[Motor] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """生成三相电流原始数据。

    两种模式：
      · 标幺模式（motor=None，默认）：维持 v0.1 行为，幅值单位为标幺。
        load_profile 第三项 = 幅值倍率；faults 里的 stall/load_step 为 "×depth"。
      · 物理模式（motor 给定，如 5kW 电机）：幅值单位为真实安培(A)。
        load_profile 第三项 = 负载转矩(标幺 0~1)，由电机模型换算成相电流；
        faults 里的 stall/load_step/unbalance 按物理量解释：
          stall     → 电流跳到堵转电流（≈ stall_current_pu × 额定）
          load_step → 电流跳到给定转矩对应的电流（depth = 转矩标幺）
          unbalance → 电流不平衡度（depth = 不平衡分数，B降C升）

    Args:
        fs: 采样率 (SPS/通道)
        duration: 时长 (s)
        f1: 基波频率 (Hz，变频器输出频率)
        amplitude: 标幺模式基波幅值（物理模式忽略，改用电机参数）
        harmonics: {谐波次数: 相对基波幅值}
        fsw: 变频器开关频率 (Hz)
        pwm_depth: PWM 纹波相对幅值
        snr_db: 信噪比 (dB)，inf 表示无噪声
        adc_bits: 量化位深
        full_scale: 量化满量程；None 时 标幺模式=2.0，物理模式=2×信号峰值(自动量程)
        imbalance: 三相不平衡度（B 相 +imbalance，C 相 -imbalance）
        load_profile: [(t0, t1, val), ...] 包络；标幺=幅值倍率，物理=负载转矩(标幺)
        faults: Fault 列表
        seed: 随机种子
        motor: 电机物理参数；给定则进入物理(安培)模式

    Returns:
        (signal, timestamps_ns, metadata)
          signal: (N, 3) float —— 三相电流（标幺/安培）
          timestamps_ns: (N,) int64 —— 模拟 UTC 时戳（步长 1e9/fs ns）
          metadata: dict —— 契约参数 + 量化/削波统计 + 电机参数(物理模式)
    """
    if harmonics is None:
        harmonics = {5: 0.12, 7: 0.06, 11: 0.03, 13: 0.02}  # 电机特征谐波
    if faults is None:
        faults = []

    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = _make_t(fs, duration)

    physical = motor is not None

    # 负载包络（过程状态：启动/空载/负载）
    if load_profile is None:
        env = np.ones(n)
    else:
        env = np.zeros(n)
        for t0, t1, a in load_profile:
            i0, i1 = int(t0 * fs), int(t1 * fs)
            env[i0:i1] = a
        # 标幺模式：未被 load_profile 覆盖的区段视为满载(=1.0)
        # 物理模式：不覆盖——env==0 表示空载转矩 0.0，是合法值，不能改写
        if not physical:
            env[env == 0] = 1.0

    # 基波峰值包络 amp_env：标幺模式 = amplitude×env；物理模式 = √2×I(T)
    if physical:
        amp_env = np.array([motor.peak_current_a(T) for T in env])
        amp_nominal = float(motor.peak_current_a(1.0))   # 额定负载峰值电流
    else:
        amp_env = amplitude * env
        amp_nominal = amplitude

    phase_shift = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])  # A/B/C
    imb = np.array([1.0, 1.0 + imbalance, 1.0 - imbalance])

    signal = np.zeros((n, 3))
    for ch, ph in enumerate(phase_shift):
        s = amp_env * np.sin(2.0 * np.pi * f1 * t + ph)
        for k, a in harmonics.items():
            s += amp_env * a * np.sin(2.0 * np.pi * k * f1 * t + k * ph + 0.5)
        signal[:, ch] = s * imb[ch]

    # PWM 开关纹波（相对额定基波幅值）
    for ch in range(3):
        signal[:, ch] += pwm_depth * amp_nominal * np.sin(2.0 * np.pi * fsw * t + ch)

    # 高斯噪声
    if snr_db < np.inf:
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_std = signal_rms / (10.0 ** (snr_db / 20.0))
        signal += rng.normal(0.0, noise_std, signal.shape)

    # 施加故障（物理模式：stall/load_step/unbalance 按真实电流/不平衡解释）
    fault_amp = amp_nominal if physical else amplitude
    for f in faults:
        signal = _apply_fault(signal, t, f, fs, f1, fault_amp, rng, motor=motor)

    # 时戳（模拟 UTC）
    timestamps_ns = np.arange(n, dtype=np.int64) * int(1e9 / fs)

    # 量化 + 削波统计（物理模式默认按信号峰值自动量程，防整段削波）
    if full_scale is None:
        full_scale = 2.0 * float(np.max(np.abs(signal))) if physical else 2.0
    signal_q, clip = _quantize(signal, adc_bits, full_scale, return_clip=True)

    meta = {
        "fs": fs,
        "duration": duration,
        "f1": f1,
        "adc_bits": adc_bits,
        "full_scale": full_scale,
        "snr_db": snr_db,
        "channels": 3,
        "sync": True,
        "clip_ratio": float(clip.mean()),
        "units": "A (real)" if physical else "A (per-unit)",
        "faults": [f.kind for f in faults],
    }
    if physical:
        meta["motor"] = {
            "rated_power_w": motor.rated_power_w,
            "rated_voltage_v": motor.rated_voltage_v,
            "frequency_hz": motor.frequency_hz,
            "poles": motor.poles,
            "rated_current_a": round(motor.rated_current_a, 3),
            "no_load_current_a": round(motor.no_load_current_a, 3),
            "stall_current_a": round(motor.stall_current_a, 3),
        }
    return signal_q, timestamps_ns, meta


# ============================================================
# 电流 + 电压 联合生成（交叉验证）
# ============================================================

def generate_vi_dataset(
    fs: float = 16000.0,
    duration: float = 10.0,
    f1: float = 50.0,
    motor: Optional[Motor] = None,
    harmonics: Optional[Dict[int, float]] = None,
    fsw: float = 8000.0,
    pwm_depth: float = 0.03,
    snr_db: float = 60.0,
    adc_bits: int = 24,
    full_scale: Optional[float] = None,
    imbalance: float = 0.0,
    load_profile: Optional[List[Tuple[float, float, float]]] = None,
    faults: Optional[List[Fault]] = None,
    seed: int = 42,
    voltage_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """生成三相电流 + 对齐的三相电压（电流/电压交叉验证用）。

    物理模型：
      · 电流与电压同频，相位差 φ = arccos(PF(负载))，电压超前电流（感性负载）
      · 空载 PF≈0.20（励磁无功为主），满载 PF≈额定 0.83
      · 电压默认额定 380V（相 220V）；一阶耦合：voltage_scale 同时缩放 V 与 I，
        使电压跌落时阻抗 Z=V/I 基本不变（供电侧特征）；电机侧故障(堵转等)
        只改电流 → Z 骤降（电机侧特征）

    Args:
        同 generate_dataset（motor 必给），另：
        voltage_scale: 电压幅值系数（1.0 = 额定；0.85 = 15% 电压跌落等供电侧问题）

    Returns:
        (current, voltage, timestamps_ns, metadata)
          current: (N,3) 三相电流 (A)
          voltage: (N,3) 三相相电压 (V)
          timestamps_ns: (N,) int64
          metadata: dict
    """
    if motor is None:
        raise ValueError("generate_vi_dataset 需要传入 motor（如 Motor() 5kW）")

    sig, ts, meta = generate_dataset(
        fs=fs, duration=duration, f1=f1, motor=motor,
        harmonics=harmonics, fsw=fsw, pwm_depth=pwm_depth,
        snr_db=snr_db, adc_bits=adc_bits, full_scale=full_scale,
        imbalance=imbalance, load_profile=load_profile, faults=faults, seed=seed,
    )
    n = int(duration * fs)
    t = _make_t(fs, duration)

    # 负载转矩包络（物理模式：load_profile 第三项 = 转矩标幺，0=空载）
    if load_profile is None:
        env = np.ones(n)
    else:
        env = np.zeros(n)
        for t0, t1, a in load_profile:
            env[int(t0 * fs):int(t1 * fs)] = a

    # 相位差 φ(t) = arccos(PF(torque))，电压超前电流
    phi = np.arccos(np.clip([motor.load_power_factor(T) for T in env], 1e-6, 1.0))
    phase_shift = np.array([0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0])
    v_peak = motor.rated_voltage_peak_v * voltage_scale

    voltage = np.zeros((n, 3))
    for ch, ph in enumerate(phase_shift):
        voltage[:, ch] = v_peak * np.sin(2.0 * np.pi * f1 * t + ph + phi)

    v_full = full_scale if full_scale is not None else 2.0 * v_peak
    vq, _ = _quantize(voltage, adc_bits, v_full)

    # 一阶耦合：电压跌落时电流跟随（I ∝ V，固定阻抗），保持 Z≈恒定（供电侧特征）
    if voltage_scale != 1.0:
        sig = sig * voltage_scale

    meta["voltage"] = {
        "rated_line_v": motor.rated_voltage_v,
        "rated_phase_v": round(motor.rated_voltage_phase_v, 2),
        "voltage_scale": voltage_scale,
        "units": "V (phase, real)",
    }
    return sig, vq, ts, meta


# ============================================================
# 演示 / 自检
# ============================================================

def _demo() -> None:
    """生成一份正常 + 一份含多故障的样本，打印统计并落盘。"""
    import os

    os.makedirs("sim", exist_ok=True)

    # 正常样本（带负载周期）
    sig, ts, meta = generate_dataset(
        duration=10.0,
        f1=50.0,
        load_profile=[(0.0, 2.0, 0.3), (2.0, 8.0, 1.0), (8.0, 10.0, 0.3)],
    )
    np.save("sim/abc_normal.npy", sig)
    print(f"[正常] {sig.shape}, Fs={meta['fs']}, 削波={meta['clip_ratio']:.4f}")

    # 多故障样本（ML 打底用）
    sig2, ts2, meta2 = generate_dataset(
        duration=20.0,
        f1=50.0,
        faults=[
            Fault(kind="load_step", start=3.0, dur=0.5, depth=1.6),
            Fault(kind="rotor_sideband", start=6.0, dur=4.0, slip=0.03, depth=0.02),
            Fault(kind="spike", start=11.0, dur=2.0, depth=0.5, params={"density": 0.002}),
            Fault(kind="stall", start=15.0, dur=1.0, depth=1.8),
        ],
    )
    np.save("sim/abc_faults.npy", sig2)
    print(f"[多故障] {sig2.shape}, 故障={meta2['faults']}")

    print("已保存 sim/abc_normal.npy, sim/abc_faults.npy")


def _demo_motor() -> None:
    """5kW 电机物理模式演示：堵转 / 负载突变 / 三相不平衡（真实安培）。"""
    motor = Motor()  # 5kW / 380V / 50Hz / 4 极
    fs = 16000.0
    print("=" * 64)
    print("5kW 电机物理模式 — 真实电流演示 (stall / load_step / unbalance)")
    print("=" * 64)
    print(f"  额定电流 ≈ {motor.rated_current_a:.2f} A")
    print(f"  空载电流 ≈ {motor.no_load_current_a:.2f} A")
    print(f"  堵转电流 ≈ {motor.stall_current_a:.2f} A ({motor.stall_current_pu:.0f}×额定)")

    sig, ts, meta = generate_dataset(
        duration=10.0, f1=50.0, motor=motor,
        load_profile=[(0.0, 2.0, 0.0), (2.0, 7.0, 1.0),
                      (7.0, 9.0, 1.0), (9.0, 10.0, 0.0)],
        faults=[
            Fault(kind="load_step", start=2.0, dur=0.3, depth=1.0),
            Fault(kind="stall", start=4.0, dur=1.0),
            Fault(kind="unbalance", start=7.0, dur=2.0, depth=0.05),
        ],
    )

    def _rms(a: float, b: float) -> float:
        return float(np.sqrt(np.mean(sig[int(a * fs):int(b * fs)] ** 2)))

    print(f"  0-2s 空载 RMS       ≈ {_rms(0, 2):6.2f} A  (预期 {motor.no_load_current_a:.2f})")
    print(f"  2-3s 负载突变 RMS   ≈ {_rms(2, 3):6.2f} A  (预期 {motor.rated_current_a:.2f})")
    print(f"  4-5s 堵转 RMS       ≈ {_rms(4, 5):6.2f} A  (预期 {motor.stall_current_a:.2f})")
    rb = float(np.sqrt(np.mean(sig[7 * int(fs):9 * int(fs), 1] ** 2)))
    rc = float(np.sqrt(np.mean(sig[7 * int(fs):9 * int(fs), 2] ** 2)))
    print(f"  7-9s 不平衡 B/C 相 RMS ≈ {rb:.2f} / {rc:.2f} A  (5% 不平衡，B降C升)")
    print(f"  单位: {meta['units']}, 削波率: {meta['clip_ratio']:.4f}")


if __name__ == "__main__":
    _demo()
    _demo_motor()
