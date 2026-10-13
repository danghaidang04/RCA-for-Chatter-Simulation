"""Sinh K "máy" mô phỏng: mỗi máy có MỘT bộ tham số DDE cố định (pj: hệ số nhân của f_n, ζ, k, K_t, K_e; cùng khoảng như bộ sinh lệch của cutting_sim)
nên mô hình tác động của lỗi khác nhau giữa các máy nhưng ổn định trong một máy. Cùng nền khỏe thật cho mọi máy (chỉ khác dấu hiệu lỗi).
Ghi data/machines.npz (F: (K, rows, 6, 30), pj, split). Chạy: python make_machines.py (vài chục phút, chạy nền được)."""
import json, os, sys
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CS = os.path.join(ROOT, "cutting_sim"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CS); sys.path.insert(0, ROOT); os.chdir(CS)
K = int(os.environ.get("N_MACHINES", "6")); R = 3


def machine_pj(mid):
    r = np.random.default_rng(5000 + mid); lu = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))
    return dict(fn=lu(0.75, 1.33), zeta=lu(0.5, 2.0), k=lu(0.6, 1.7), Kt=lu(0.6, 1.6), Ke=lu(0.4, 2.5))


def make_row_fixed(args):
    """Giống rca_data.make_row_lat (part='full') nhưng pj cố định theo máy."""
    import rca_data as rd
    mrow, seed, u, pj = args; g = rd._G; rng = np.random.default_rng(seed)
    d0 = rd.dde_signals(mrow, "NORMAL", 1.0, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], pj); out = []
    for c in rd.CLASSES:
        b = g["gen"].signals(mrow, u, rng, g["n_a"], g["n_m"])
        if c == "NORMAL": sig = b
        else:
            lo, hi = rd.SEV[c]; sev = (rng.choice([-1, 1]) * rng.uniform(0.03, 0.15) + 1.0) if c == "RPM_MISMATCH" else rng.uniform(lo, hi)
            d1 = rd.dde_signals(mrow, c, sev, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], pj); n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
            sig = (b[0][:n[0]] + g["gacc"] * (d1[0][:n[0]] - d0[0][:n[0]]), b[1][:n[0]] + g["gacc"] * (d1[1][:n[0]] - d0[1][:n[0]]), b[2][:n[1]] + g["gmic"] * (d1[2][:n[1]] - d0[2][:n[1]]))
        out.append(rd.rca_features(sig[0], sig[1], sig[2], g["fs_a"], g["sr"], mrow))
    return np.array(out)


def main():
    import rca_data as rd
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json")); cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid)
    sim = np.load(os.path.join(HERE, "data", "sim_sets.npz"), allow_pickle=True); split = sim["split"]; row_cut = sim["row_cut"]       # cùng cách tách đường cắt với các thí nghiệm khác
    assert len(split) == len(meta) * R
    Fs, pjs = [], []; p0 = os.path.join(HERE, "data", "machines.npz")
    if os.path.exists(p0) and os.environ.get("APPEND", "0") == "1": old = np.load(p0, allow_pickle=True); Fs = list(old["F"]); pjs = json.loads(str(old["pj"]))          # nối thêm máy mới, giữ máy cũ (cùng hạt giống nên tái lập)
    for mid in range(len(Fs), K):
        pj = machine_pj(mid); rng = np.random.default_rng(900 + mid); u_cut = {c: rng.normal(0, np.sqrt(gen.between)) for c in ucut}
        jobs = [(meta[i], 70000 * mid + 10000 * r + i, u_cut[cid[i]], pj) for r in range(R) for i in range(len(meta))]
        with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"])) as pool: out = pool.map(make_row_fixed, jobs, chunksize=8)
        Fs.append(np.array(out)); pjs.append(pj); print("machine", mid, pj, Fs[-1].shape, flush=True)
        np.savez_compressed(os.path.join(HERE, "data", "machines.npz"), F=np.array(Fs), pj=json.dumps(pjs), split=split, row_cut=row_cut)
    print("done", flush=True)


if __name__ == "__main__":
    main()
