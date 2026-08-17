"""
MCSA 新故障类型自检（00 仿真器 v0.3 新增：bearing / stator_interturn / eccentricity / phase_loss）
================================================================================================
验证合成的故障电流波形，其 FFT 频谱在**预期边带频率**出现峰值，
并与 mcp-server-mcsa 故障理论对照。

用法：
    C:\\...\\python.exe 07_测试工具\\verify_mcsa_faults.py

运行：pass 表示全部边带位置正确。
"""
import os
import sys

import numpy as np

# 保证可从仓库根目录 / 任意 cwd 导入 00 模块
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "00_数据生成与仿真"))

from current_simulator import Fault, generate_dataset  # noqa: E402
from mcsa_faults import bearing_frequencies, rotor_mech_freq_hz  # noqa: E402

FS = 16000.0
F1 = 50.0
DUR = 2.0
N = int(DUR * FS)


def top_freqs(sig: np.ndarray, n_peaks: int = 6) -> list:
    """返回信号频谱前 n 个峰值（归一化幅值降序）的频率列表。"""
    w = np.hanning(N)
    X = np.abs(np.fft.rfft(sig * w))
    X /= max(X)
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    order = np.argsort(X)[-n_peaks:][::-1]
    return sorted(round(float(freqs[i]), 2) for i in order)


def has_peak(sig: np.ndarray, target_hz: float, tol: float = 0.5) -> bool:
    """目标频率 ±tol Hz 内是否存在显著谱峰（相对基波 > 1%）。"""
    w = np.hanning(N)
    X = np.abs(np.fft.rfft(sig * w))
    X /= max(X)
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    mask = np.abs(freqs - target_hz) <= tol
    return bool(np.any(X[mask] > 0.01))


def main() -> int:
    checks = []
    # 1) 偏心：fs ± k·fr，fr = 2·50·(1-0.03)/4 = 24.25 → 25.75 / 74.25
    s, _, _ = generate_dataset(
        fs=FS, duration=DUR, f1=F1,
        faults=[Fault(kind="eccentricity", start=0.2, dur=1.6, slip=0.03,
                      depth=0.05, k=1, params={"poles": 4})],
    )
    checks.append(("eccentricity 25.75Hz", has_peak(s[:, 0], 25.75)))
    checks.append(("eccentricity 74.25Hz", has_peak(s[:, 0], 74.25)))

    # 2) 定子匝间短路：fs ± 2k·fr → 50 ± 48.5 → 1.5 / 98.5
    s, _, _ = generate_dataset(
        fs=FS, duration=DUR, f1=F1,
        faults=[Fault(kind="stator_interturn", start=0.2, dur=1.6, slip=0.03,
                      depth=0.05, k=1, params={"poles": 4})],
    )
    checks.append(("stator_interturn 98.5Hz", has_peak(s[:, 0], 98.5)))

    # 3) 轴承外圈：fr=24.25，BPFO=0.5·8·24.25·(1-0.006/0.028)=76.2 → 50±76.2 → 126.2
    fr = rotor_mech_freq_hz(F1, 0.03, 4)
    geom = bearing_frequencies(fr_rot_hz=fr, n_balls=8, ball_d=0.006, pitch_d=0.028,
                               contact_angle_deg=0.0)
    s, _, _ = generate_dataset(
        fs=FS, duration=DUR, f1=F1,
        faults=[Fault(kind="bearing", start=0.2, dur=1.6, slip=0.03, depth=0.05,
                      k=1, params={"n_balls": 8, "ball_d": 0.006, "pitch_d": 0.028,
                                   "contact_angle_deg": 0.0, "ring": "outer",
                                   "poles": 4})],
    )
    checks.append(("bearing BPFO 边带 ~126.2Hz",
                   has_peak(s[:, 0], F1 + geom["bpfo"])))

    # 4) 缺相：A 相电流趋零，B/C 保持
    s, _, _ = generate_dataset(
        fs=FS, duration=DUR, f1=F1,
        faults=[Fault(kind="phase_loss", start=0.2, dur=1.6, depth=0.0,
                      params={"phase": 0, "residual": 0.05})],
    )
    seg = slice(int(0.2 * FS), int(1.8 * FS))
    rms_a = float(np.sqrt(np.mean(s[seg, 0] ** 2)))
    rms_b = float(np.sqrt(np.mean(s[seg, 1] ** 2)))
    checks.append(("phase_loss A 相趋零", rms_a < 0.02 * rms_b))

    # 5) 回归：已有故障类型仍可用
    s, _, _ = generate_dataset(
        fs=FS, duration=DUR, f1=F1,
        faults=[Fault(kind="rotor_sideband", start=0.2, dur=1.6, slip=0.03,
                      depth=0.05, k=1)],
    )
    checks.append(("rotor_sideband 回归 (50±3Hz 边带)", has_peak(s[:, 0], 47.0)))

    # 输出
    ok = True
    print("MCSA 故障合成自检")
    print("-" * 46)
    for name, passed in checks:
        mark = "PASS" if passed else "FAIL"
        ok = ok and passed
        print(f"  [{mark}] {name}")
    print("-" * 46)
    print("结论:", "全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
