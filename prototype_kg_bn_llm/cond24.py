"""Mô hình XẾP HẠNG của ứng dụng: Bayes Gauss có điều kiện trên 24 đặc trưng phổ (4 chỉ số + 10 băng gia tốc + 10 băng micro), nối đầy đủ cause -> đặc trưng.
KG dùng để TRUY VẾT/giải thích ở mức NHÓM băng tần (cộng dồn đóng góp các băng trong nhóm); nó không ràng buộc mô hình xếp hạng này (xem README). Ghi report/cond24_results_k10.json."""
import json, os
import numpy as np
import cond_experiment as ce
import experiment as ex
from bn import CondBN, ece
from kg import CAUSES

HERE = os.path.dirname(os.path.abspath(__file__)); SUF = "_k10"
FEATS = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic"] + [f"accB{i}" for i in range(10)] + [f"micB{i}" for i in range(10)]
GROUP = {"log10_acc": "log10_acc", "mic_dB": "mic_dB", "chatter_acc": "chatter_acc", "chatter_mic": "chatter_mic"}
for i in range(10):
    GROUP[f"accB{i}"] = "acc_low" if i <= 2 else ("acc_mid" if i == 3 else ("acc_res" if i <= 6 else "acc_high"))      # cùng cách gộp với bn.to_nodes: low = B0-2, res = B4-6, high = B7-9; B3 chưa có nút KG
    GROUP[f"micB{i}"] = "mic_low" if i <= 2 else ("mic_high" if i >= 6 else "mic_mid")
GROUPS = list(dict.fromkeys(GROUP[f] for f in FEATS))


def load():
    d = np.load(os.path.join(HERE, "data", f"sim_sets{SUF}.npz"), allow_pickle=True); split = d["split"]; F, Fs = d["F_nom"][..., :24], d["F_shift"][..., :24]
    meta = np.load(os.path.join(os.path.dirname(HERE), "cutting_sim", "output", "real_windows.npz"))["meta"]; idx = np.arange(len(split)) % len(meta); A = ex.cond_design(meta[idx, 5], meta[idx, 6], meta[idx, 7], meta[idx, 4])
    key = np.array([f"{meta[i, 5]:.0f}|{meta[i, 6]}|{meta[i, 4]:.0f}" for i in idx]); sd0 = F[split == "train", 0].std(0); return F / sd0, Fs / sd0, A, key, split, sd0, d["row_cut"]


def main():
    F, Fs, A, key, split, sd0, row_cut = load(); tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]; te = np.where(split == "test")[0]; gva, gte = ce.groups_of(va, key, 1), ce.groups_of(te, key)
    mod = CondBN().fit(np.repeat(A[tr], 6, axis=0), F[tr].reshape(-1, 24), np.tile(np.arange(6), len(tr)))
    Dv, Av, Lv = ce.draw_same_condition(F, A, gva, 120, np.random.default_rng(5)); Dn, An, Ln = ce.draw_same_condition(F, A, gte, 200, np.random.default_rng(6)); Ds, As_, Ls = ce.draw_same_condition(Fs, A, gte, 200, np.random.default_rng(7))
    ce.fit_T(mod, Dv, Av, Lv); nom = ce.evaluate(mod, Dn, An, Ln); sh = ce.evaluate(mod, Ds, As_, Ls); nT1 = ce.evaluate(mod, Dn, An, Ln, T=1.0)
    print(f"24-feature condition-aware Bayes: T={mod.T:.1f} nominal top1 {nom['top1']:.3f} ece {nom['ece']:.3f} per class {np.round(nom['per_class'], 2).tolist()} | shifted top1 {sh['top1']:.3f} ece {sh['ece']:.3f} | ece at T=1: {nT1['ece']:.3f}")
    best = lambda d, a: int(mod.at(a).posterior(d - mod.baseline(a))[1].argmax())
    def d2(d, a):
        mm = mod.at(a); c = best(d, a); x = (d - mod.baseline(a)).mean(0) - mm.mu[c]; return float(len(d) * x @ mm.P @ x), x
    ref = [d2(d, a) for d, a in zip(Dn, An)]; q95 = float(np.percentile([r[0] for r in ref], 95)); shf = [d2(d, a)[0] for d, a in zip(Ds, As_)]; node_sd = np.std([r[1] for r in ref], axis=0)
    # trọng số cạnh theo NHÓM băng tần (trung bình các băng trong nhóm) + khoảng tin cậy bootstrap theo đường cắt
    ucut = np.unique(row_cut[tr]); by = {c: tr[row_cut[tr] == c] for c in ucut}; brng = np.random.default_rng(11); Wb = []; gi = {g: [i for i, f in enumerate(FEATS) if GROUP[f] == g] for g in GROUPS}
    for _ in range(100):
        rows = np.concatenate([by[c] for c in brng.choice(ucut, len(ucut))]); m_ = CondBN().fit(np.repeat(A[rows], 6, axis=0), F[rows].reshape(-1, 24), np.tile(np.arange(6), len(rows)))
        W = np.einsum("cpk,p->ck", m_.beta[1:] - m_.beta[0], A[rows].mean(0)) / m_.sd; Wb.append(np.stack([W[:, gi[g]].mean(1) for g in GROUPS], axis=1))
    Wb = np.array(Wb); wm, lo, hi = Wb.mean(0), np.percentile(Wb, 2.5, axis=0), np.percentile(Wb, 97.5, axis=0)
    out = {"nominal": nom, "shifted": sh, "ece_without_temperature": nT1["ece"], "feats": FEATS, "groups": GROUPS, "group_of": GROUP, "misfit": {"q95_nominal": q95, "flag_rate_nominal": float(np.mean([r[0] > q95 for r in ref])), "flag_rate_shifted": float(np.mean([x > q95 for x in shf])), "node_sd_ref": node_sd.tolist()},
           "edges_table": {f"{c}|{g}": {"weight": float(wm[i, k]), "ci_lo": float(lo[i, k]), "ci_hi": float(hi[i, k])} for i, c in enumerate(CAUSES) for k, g in enumerate(GROUPS)}, "model": {"beta": mod.beta.tolist(), "Sigma": mod.Sigma.tolist(), "T": mod.T, "sd0": sd0.tolist()}}
    print("misfit q95", round(q95, 1), "flag rate nominal/shifted", round(out["misfit"]["flag_rate_nominal"], 3), round(out["misfit"]["flag_rate_shifted"], 3))
    json.dump(out, open(os.path.join(HERE, "report", f"cond24_results{SUF}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
