"""Kiểm tra: cơ chế dao sứt rãnh (mất đối xứng răng = độ lệch tâm hiệu dụng lớn) có tái tạo dấu hiệu dao hỏng THẬT không?
So mẫu lệch (đơn vị sd khỏe mô phỏng) của 9 nút cảm biến giữa 'sứt rãnh' và 'khỏe' với mẫu lệch thật của D* (output/real_defect_check.json)."""
import json
from multiprocessing import Pool
import numpy as np
import rca_data as rd
import rca_causal as rc

_J = {}


def init(gen, fs_a, sr, n_a, n_m):
    rd._init(gen, fs_a, sr, n_a, n_m, 0.0, 0.0); _J.update(gen=gen, fs_a=fs_a, sr=sr, n_a=n_a, n_m=n_m)


def job(args):
    mrow, sev_um, seed = args; g = rd._G; rng = np.random.default_rng(seed); u = np.zeros(12)
    b = g["gen"].signals(mrow, u, rng, g["n_a"], g["n_m"]); d0 = rd.dde_signals(mrow, "NORMAL", 1.0, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], None)
    d1 = rd.dde_signals(mrow, "TOOL_DEFECT", sev_um, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], None); n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
    return (b[0][:n[0]], b[1][:n[0]], b[2][:n[1]]), ([x[:n[0]] for x in (d1[0] - d0[0][:len(d1[0])], d1[1] - d0[1][:len(d1[1])])] + [d1[2][:n[1]] - d0[2][:n[1]]]), mrow


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); gain = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    real = json.load(open(f"{rd.OUT}/real_defect_check.json")); rv = np.array([real["real_shift"][n] for n in rc.SENS]); idx = np.where(meta[:, 3] == 0)[0][::2][:72]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    with Pool(10, initializer=init, initargs=(gen, fs_a, sr, n_a, n_m)) as pool:
        res = {}
        for sev in (10, 20, 40, 80):
            out = pool.map(job, [(meta[i], float(sev), 5000 + k) for k, i in enumerate(idx)], chunksize=4)
            for gs in (1, 10, 100):
                ga, gm = gain["g_acc"] * gs, gain["g_mic"] * gs; H, D = [], []
                for (b, dd, mrow) in out:
                    n = min(len(b[0]), len(dd[0])), min(len(b[2]), len(dd[2]))
                    H.append(rd.rca_features(b[0], b[1], b[2], fs_a, sr, mrow)); D.append(rd.rca_features(b[0][:n[0]] + ga * dd[0][:n[0]], b[1][:n[0]] + ga * dd[1][:n[0]], b[2][:n[1]] + gm * dd[2][:n[1]], fs_a, sr, mrow))
                Hn, Dn = rc.sens_nodes(np.array(H)), rc.sens_nodes(np.array(D)); v = (Dn.mean(0) - Hn.mean(0)) / (Hn.std(0) + 1e-9); res[f"runout{sev}um_gain{gs}x"] = {"shift": dict(zip(rc.SENS, v.tolist())), "cosine": cos(v, rv)}
                print(f"runout {sev:3d} µm, gain x{gs:<3d}: cosine={cos(v, rv):+.2f} | " + " ".join(f"{n[:8]}={x:+.1f}" for n, x in zip(rc.SENS, v)), flush=True)
    print("\nthật (D*):      " + " ".join(f"{n[:8]}={x:+.1f}" for n, x in zip(rc.SENS, rv)))
    json.dump(res, open(f"{rd.OUT}/defect_mechanism_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
