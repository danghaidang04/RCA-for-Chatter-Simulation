"""Chuỗi 4 cảm biến (khối LTI kiểu Simulink bằng scipy.signal) từ kết quả mô phỏng DDE.

 1. Relative tool-work vibration : cảm biến dịch chuyển không tiếp xúc (x,y) [µm], BW ~8 kHz
 2. Accelerometer                : gia tốc đầu dao/ổ gá (ax, ay) [g], band 5 Hz-10 kHz, nhiễu IEPE
 3. Microphone                   : áp suất âm p [Pa] ~ (ρ0 A / 4π r)·a(t-r/c) + ồn cắt ∝ dF/dt + ồn nền
 4. Force dynamometer            : lực phản lực Fx,Fy [N] qua động học bàn đo (2nd-order, fn_d ~ 2 kHz) + trôi + nhiễu
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

G = 9.81


@dataclass(frozen=True)
class SensorConfig:
    acc_gain: float = 1.0         # hệ số truyền từ gia tốc đầu dao -> gia tốc tại vị trí cảm biến (thân trục chính)
    mic_gain: float = 1.0
    disp_noise_um: float = 0.05
    disp_bw: float = 8000.0
    acc_noise_g: float = 0.01
    acc_rpm_exp: float = 0.0      # nhiễu accelerometer ∝ (rpm/10000)^exp (ổ trục, khí động)
    mic_rpm_exp: float = 0.0
    acc_band: tuple = (5.0, 10000.0)
    mic_area: float = 2e-3        # m², diện tích bức xạ hiệu dụng
    mic_dist: float = 0.30        # m
    mic_noise_pa: float = 0.01    # ~54 dB SPL nền xưởng
    mic_cut_coeff: float = 2e-7   # Pa/(N/s): ồn cắt trực tiếp
    dyno_fn: float = 2000.0       # Hz (bàn đo + phôi)
    dyno_zeta: float = 0.03
    dyno_noise_n: float = 0.5
    dyno_drift_n: float = 1.0


def _lowpass(x, fs, fc, order=2):
    sos = signal.butter(order, fc / (fs / 2), "low", output="sos")
    return signal.sosfilt(sos, x)


def _bandpass(x, fs, lo, hi, order=2):
    sos = signal.butter(order, [lo / (fs / 2), min(hi, 0.45 * fs) / (fs / 2)], "band", output="sos")
    return signal.sosfilt(sos, x)


def _second_diff(q, fs):
    a = np.zeros_like(q)
    a[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) * fs ** 2
    a[0], a[-1] = a[1], a[-2]
    return a


def apply_sensors(sim, cfg=SensorConfig(), seed=0):
    rng = np.random.default_rng(seed + 7919)
    fs, t, n = sim["fs"], sim["t"], len(sim["t"])
    rr = sim.get("rpm", 1e4) / 1e4
    out = {"t": t, "fs": fs}

    # 1) Relative tool-work vibration (µm)
    for ax in ("x", "y"):
        v = _lowpass(sim[ax] * 1e6, fs, cfg.disp_bw) + rng.normal(0, cfg.disp_noise_um, n)
        out[f"vib_{ax}"] = v

    # 2) Accelerometer (g)
    acc = {}
    for ax in ("x", "y"):
        a = _second_diff(sim[ax], fs) / G
        acc[ax] = a
        out[f"acc_{ax}"] = cfg.acc_gain * _bandpass(a, fs, *cfg.acc_band) + rng.normal(0, cfg.acc_noise_g * rr ** cfg.acc_rpm_exp, n)

    # 3) Microphone (Pa): bức xạ do gia tốc + ồn cắt + ồn nền; trễ truyền âm r/c
    a_res = np.hypot(acc["x"], acc["y"]) * G * np.sign(acc["x"] + 1e-30)  # m/s² (có dấu theo x)
    p_rad = 1.2 * cfg.mic_area / (4 * np.pi * cfg.mic_dist) * a_res
    dF = np.gradient(sim["Fx"], 1 / fs)
    p_cut = cfg.mic_cut_coeff * dF
    delay = int(round(cfg.mic_dist / 343.0 * fs))
    p = np.roll(p_rad + p_cut, delay); p[:delay] = 0.0
    p = cfg.mic_gain * _bandpass(p, fs, 20.0, 10000.0) + rng.normal(0, cfg.mic_noise_pa * rr ** cfg.mic_rpm_exp, n)
    out["mic"] = p

    # 4) Force dynamometer: phản lực lên phôi (-F_tool) qua hàm truyền bậc 2 + trôi + nhiễu
    wd = 2 * np.pi * cfg.dyno_fn
    num, den = [wd ** 2], [1, 2 * cfg.dyno_zeta * wd, wd ** 2]
    bz, az = signal.bilinear(num, den, fs)
    drift = np.cumsum(rng.normal(0, 1, n)); drift = cfg.dyno_drift_n * (drift - drift.mean()) / (np.abs(drift).max() + 1e-12)
    for ax in ("x", "y"):
        f = signal.lfilter(bz, az, -sim[f"F{ax}"])
        out[f"F_{ax}"] = f + drift + rng.normal(0, cfg.dyno_noise_n, n)
    return out
