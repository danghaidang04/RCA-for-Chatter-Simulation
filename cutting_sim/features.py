"""Trích đặc trưng từ 4 cảm biến (dùng cho RCA và hiệu chỉnh)."""
import numpy as np
from scipy.fft import rfft, rfftfreq

P_REF = 20e-6


def spectrum(x, fs, window=True):
    x = x - np.mean(x)
    w = np.hanning(len(x)) if window else np.ones(len(x))
    X = np.abs(rfft(x * w)) / (w.sum() / 2)
    return rfftfreq(len(x), 1 / fs), X


def nonsync_ratio(x, fs, rpm, N, f_lo=100.0, f_hi=5000.0, tol=6.0):
    """Tỷ lệ năng lượng phổ KHÔNG nằm trên các hài của tần số quay trục chính (±tol Hz) => chỉ báo chatter."""
    f, X = spectrum(x, fs)
    band = (f >= f_lo) & (f <= f_hi)
    f0 = rpm / 60.0
    k = np.round(f / f0)
    near = np.abs(f - k * f0) <= tol
    e_all = np.sum(X[band] ** 2) + 1e-30
    return float(np.sum(X[band & ~near] ** 2) / e_all)


def dominant_freq(x, fs, fmin=100.0, fmax=5000.0):
    f, X = spectrum(x, fs)
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][np.argmax(X[m])])


def extract(sig, rpm, N, t_skip=0.4):
    fs = sig["fs"]
    i0 = int(t_skip * len(sig["t"]))
    seg = lambda k: sig[k][i0:]
    out = {}
    vib = np.hypot(seg("vib_x") - seg("vib_x").mean(), seg("vib_y") - seg("vib_y").mean())
    out["vib_rms_um"] = float(np.sqrt(np.mean(vib ** 2)))
    out["vib_pp_um"] = float(np.ptp(seg("vib_x")))
    out["acc_rms_g"] = float(np.sqrt(np.mean(seg("acc_x") ** 2 + seg("acc_y") ** 2)))
    out["mic_rms_pa"] = float(np.sqrt(np.mean(seg("mic") ** 2)))
    out["mic_spl_db"] = float(20 * np.log10(out["mic_rms_pa"] / P_REF))
    Fx, Fy = seg("F_x"), seg("F_y")
    Fr = np.hypot(Fx, Fy)
    out["F_mean_n"] = float(np.mean(Fr))
    out["F_rms_dyn_n"] = float(np.sqrt(np.mean((Fx - Fx.mean()) ** 2 + (Fy - Fy.mean()) ** 2)))
    out["F_peak_n"] = float(np.max(Fr))
    out["chatter_vib"] = nonsync_ratio(seg("vib_x"), fs, rpm, N)
    out["chatter_acc"] = nonsync_ratio(seg("acc_x"), fs, rpm, N)
    out["chatter_mic"] = nonsync_ratio(seg("mic"), fs, rpm, N)
    out["chatter_force"] = nonsync_ratio(seg("F_x"), fs, rpm, N)
    out["f_dom_vib"] = dominant_freq(seg("vib_x"), fs)
    out["f_dom_mic"] = dominant_freq(seg("mic"), fs)
    out["f_res_acc"] = resonance_peak(seg("acc_y"), fs, rpm)
    out["f_res_vib"] = resonance_peak(seg("vib_y"), fs, rpm)
    return out


def resonance_peak(x, fs, rpm, lo=600.0, hi=950.0, tol=6.0):
    """Trọng tâm phổ (theo |X|²) của thành phần KHÔNG đồng bộ trong 600-950 Hz -> ước lượng tần số cộng hưởng kết cấu
    (trọng tâm ổn định hơn đỉnh đơn vì đỉnh nhảy giữa các mode lân cận)."""
    f, X = spectrum(x, fs)
    f0 = rpm / 60.0
    ok = (f >= lo) & (f <= hi) & (np.abs(f - np.round(f / f0) * f0) > tol)
    w = X[ok] ** 2
    return float(np.sum(f[ok] * w) / (np.sum(w) + 1e-30))
