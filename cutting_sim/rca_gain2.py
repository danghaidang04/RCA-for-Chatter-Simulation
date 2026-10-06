"""Hiệu chuẩn hệ số khuếch đại theo ĐỘ LỚN thay đổi: chọn (g_acc, g_mic) sao cho mức tăng trung bình log10 gia tốc và mic do lỗi DDE
bằng mức chênh thật giữa cửa sổ bất thường (nhãn 1) và bình thường (nhãn 0)."""
import json
from multiprocessing import Pool
import numpy as np
import noise_model as nm
import rca_data as rd
from rca_gain import base_pair


def shift_vs_gain(BD, jobs, fs_a, sr, ga, gm):
    out = []
    for (b, d), j in zip(BD, jobs):
        m = j[0]; n = min(len(b[0]), len(d[0])), min(len(b[2]), len(d[2]))
        f0 = nm.feats(b[0][:n[0]], b[1][:n[0]], b[2][:n[1]], fs_a, sr, m[5], int(m[8]))
        f1 = nm.feats(b[0][:n[0]] + ga * d[0][:n[0]], b[1][:n[0]] + ga * d[1][:n[0]], b[2][:n[1]] + gm * d[2][:n[1]], fs_a, sr, m[5], int(m[8]))
        out.append(f1[:2] - f0[:2])
    return np.mean(out, axis=0)


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr)
    r1 = np.where(meta[:, 3] == 1)[0]; r0 = np.where(meta[:, 3] == 0)[0]
    Fr = lambda idx: np.array([nm.feats(d["acc0"][i], d["acc1"][i], d["mic"][i], fs_a, sr, meta[i, 5], int(meta[i, 8])) for i in idx])
    t_acc, t_mic = (Fr(r1).mean(0) - Fr(r0).mean(0))[:2]; print(f"mức chênh thật (nhãn1 - nhãn0): log10 acc {t_acc:+.3f}, mic {t_mic:+.2f} dB", flush=True)
    faults = [c for c in rd.CLASSES if c != "NORMAL"]; jobs = [(meta[i], c, 777 * r + 10 * int(i) + k, np.zeros(12)) for r in range(2) for i in r1 for k, c in enumerate(faults)]
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, 0.0, 0.0)) as pool:
        BD = pool.map(base_pair, jobs, chunksize=8)
    ga_grid = 10.0 ** np.arange(-8, -3.4, 0.5); gm_grid = 10.0 ** np.arange(-6, -1.4, 0.5)
    sa = np.array([shift_vs_gain(BD, jobs, fs_a, sr, g, 0.0)[0] for g in ga_grid]); sm = np.array([shift_vs_gain(BD, jobs, fs_a, sr, 0.0, g)[1] for g in gm_grid])
    for g, s in zip(ga_grid, sa): print(f"  g_acc=1e{np.log10(g):.1f}: mức tăng log10 acc = {s:+.3f}")
    for g, s in zip(gm_grid, sm): print(f"  g_mic=1e{np.log10(g):.1f}: mức tăng mic = {s:+.2f} dB")
    interp = lambda grid, s, t: float(10 ** np.interp(t, s, np.log10(grid))) if (s.max() >= t) else float("nan")
    out = {"g_acc": interp(ga_grid, sa, t_acc), "g_mic": interp(gm_grid, sm, t_mic), "target_acc": float(t_acc), "target_mic": float(t_mic)}
    json.dump(out, open(f"{rd.OUT}/rca_gain_effect.json", "w"), indent=1); print("GAIN", out)


if __name__ == "__main__":
    main()
