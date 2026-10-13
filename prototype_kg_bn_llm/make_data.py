"""Sinh dữ liệu cho prototype (một lần): cửa sổ mô phỏng có nhãn, bộ sinh DANH ĐỊNH và bộ sinh LỆCH, tách theo ĐƯỜNG CẮT (train/val/test).
Dùng lại bộ sinh của cutting_sim (nền khỏe thật + hiệu DDE). Ghi data/sim_sets.npz. Chạy: python make_data.py (vài phút)."""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CS = os.path.join(ROOT, "cutting_sim")
sys.path.insert(0, CS); sys.path.insert(0, ROOT); os.chdir(CS)


def main():
    import rca_data as rd
    import rca_causal as rc
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr)
    g = json.load(open(f"{rd.OUT}/rca_gain_effect{rd.TAG}.json")); cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid)
    Fn, Ln = rc.gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "nominal", 11); print("nominal", Fn.shape, flush=True)
    Fs, Ls = rc.gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "shift", 22); print("shifted", Fs.shape, flush=True)
    rng = np.random.default_rng(0); by = {}; train_c, test_c = set(), set()
    for c in ucut: by.setdefault(meta[cid == c][0, 6], []).append(c)
    for ae, cs in by.items():                                       # cùng quy tắc tách như rca_causal.py: 40% đường cắt mỗi ae làm test
        cs = list(rng.permutation(cs)); nt = max(1, int(round(0.4 * len(cs)))) if len(cs) >= 3 else 0; test_c |= set(cs[:nt]); train_c |= set(cs[nt:])
    tr = sorted(train_c); rv = np.random.default_rng(1).permutation(tr); val_c = set(rv[: len(tr) // 3]); fit_c = set(tr) - val_c
    R = Fn.shape[0] // len(meta); row_cut = np.tile(cid, R)
    split = np.array(["test" if c in test_c else ("val" if c in val_c else "train") for c in row_cut])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"sim_sets{rd.TAG}.npz")
    np.savez_compressed(out, F_nom=Fn, F_shift=Fs, L_nom=Ln, L_shift=Ls, row_cut=row_cut, split=split, classes=np.array(rd.CLASSES), gain=np.array([g["g_acc"], g["g_mic"]]))
    print("saved", out, "| cuts train/val/test:", len(fit_c), len(val_c), len(test_c), "| rows", dict(zip(*np.unique(split, return_counts=True))), flush=True)


if __name__ == "__main__":
    main()
