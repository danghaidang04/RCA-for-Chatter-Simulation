"""Ablation RCA nhân quả (gỡ vòng tròn):  proxy ∈ {soft: cảm biến mềm huấn luyện có giám sát | physical: ước lượng vật lý KHÔNG dùng nhãn}
 graph ∈ {physics, shuffled (cạnh ngẫu nhiên cùng bậc), none (không cạnh)};  gain ∈ {0.3,1,3} x hệ số đã hiệu chuẩn;  3 hạt giống (cách chia + sinh dữ liệu).
Nút rpm luôn là phép đo bộ điều khiển (log N_thực/N_kế hoạch + nhiễu 0.5%), như Spindle_Speed trong mtc.csv.
Báo Top-1 (trung bình 5 nguyên nhân) ở m=10 và m=20, kịch bản in-distribution và DDE LỆCH."""
import json
from types import SimpleNamespace
import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import rca_data as rd
import rca_causal as rc

MS = (10, 20); NTR = 60


def phys_proxies(X):
    acc = X[:, 4:14]; mic = X[:, 14:24]; w = np.exp(acc[:, 4:8] * np.log(10)); idx = np.arange(4, 8)
    stiff = (w * idx).sum(1) / w.sum(1); damp = acc[:, 4:8].max(1) - acc[:, 4:8].mean(1); wear = mic[:, 0:3].mean(1); depth = mic[:, 6:10].mean(1)
    return np.c_[stiff, damp, wear, depth]


def system(graph, seed):
    s = rc.make_system(); idx = s.name_to_idx; dag = nx.DiGraph(); dag.add_nodes_from(range(s.d))
    if graph == "physics": edges = [(idx[u], idx[v]) for u, v in rc.EDGES]
    elif graph == "shuffled":
        rg = np.random.default_rng(seed); sens = [idx[x] for x in rc.SENS]; edges = []
        for p in rc.PROXY:
            deg = sum(1 for u, v in rc.EDGES if u == p); edges += [(idx[p], int(c)) for c in rg.choice(sens, size=deg, replace=False)]
    else: edges = []
    dag.add_edges_from(edges); s.dag = dag; return s


def split(meta, cid, seed):
    ucut = np.unique(cid); rng = np.random.default_rng(seed); by = {}; tr, te = set(), set()
    for c in ucut: by.setdefault(meta[cid == c][0, 6], []).append(c)
    for ae, cs in by.items():
        cs = list(rng.permutation(cs)); nt = max(1, int(round(0.4 * len(cs)))) if len(cs) >= 3 else 0; te |= set(cs[:nt]); tr |= set(cs[nt:])
    return tr, te


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g0 = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); R = rc.R; row_cut = np.tile(cid, R); flat = lambda A, m: A[m].reshape(-1, A.shape[-1]); res = {}
    variants = [("soft", "physics", 1.0), ("soft", "shuffled", 1.0), ("soft", "none", 1.0), ("physical", "physics", 1.0), ("soft", "physics", 0.3), ("soft", "physics", 3.0)]
    for seed in range(3):
        for gscale in (0.3, 1.0, 3.0):
            g = {"g_acc": g0["g_acc"] * gscale, "g_mic": g0["g_mic"] * gscale}
            Fn, Ln = rc.gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "nominal", 11 + 1000 * seed); Fs, Ls = rc.gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "shift", 22 + 1000 * seed)
            print(f"seed {seed} gain x{gscale}: sinh xong", flush=True)
            tr, te = split(meta, cid, seed); mtr = np.isin(row_cut, list(tr)); mte = np.isin(row_cut, list(te))
            Xtr, Ytr = flat(Fn, mtr), flat(Ln, mtr); gtr = np.repeat(row_cut[mtr], 6); rn = np.random.default_rng(seed + 5)
            soft = oof = None
            for proxy, graph, gs in variants:
                if gs != gscale: continue
                if proxy == "soft" and soft is None:
                    soft = [HistGradientBoostingRegressor(max_iter=80, learning_rate=0.15, random_state=0).fit(Xtr, Ytr[:, j]) for j in range(4)]; oof = np.zeros((len(Ytr), 4))
                    for a_, b_ in GroupKFold(3).split(Xtr, groups=gtr):
                        for j in range(4): oof[b_, j] = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.15, random_state=0).fit(Xtr[a_], Ytr[a_, j]).predict(Xtr[b_])
                P4tr = oof if proxy == "soft" else phys_proxies(Xtr); Ptr = np.c_[P4tr, Ytr[:, 4] + rn.normal(0, 0.005, len(Ytr))]
                D_obs = np.c_[Ptr, rc.sens_nodes(Xtr)][np.tile(np.arange(6), mtr.sum()) == 0]; MU, SD = D_obs.mean(0), D_obs.std(0) + 1e-9; D_obs = (D_obs - MU) / SD
                sysm = system(graph, seed); import run_full_paper_benchmark as rb
                algs = {"BRCD": rb.BRCD(sysm), "RCD": rb.RCD(sysm), "RCG": rb.RCG(sysm), "SmoothTraversal": rb.SmoothTraversal(sysm), "BARO": rb.BARO(sysm), "SimpleRCA": rb.SimpleRCA(sysm)}
                for sname, FT, LT in (("in", Fn, Ln), ("shift", Fs, Ls)):
                    Xte, Yte = flat(FT, mte), flat(LT, mte); cls = np.tile(np.arange(6), mte.sum())
                    P4 = np.c_[[s.predict(Xte) for s in soft]].T if proxy == "soft" else phys_proxies(Xte); Dte = (np.c_[np.c_[P4, Yte[:, 4] + rn.normal(0, 0.005, len(Yte))], rc.sens_nodes(Xte)] - MU) / SD
                    for m in MS:
                        acc = {a: [] for a in algs}
                        for k, f in enumerate(rc.FAULTS):
                            pf = Dte[cls == k + 1]; tn = sysm.fault_to_node[f]
                            for t in range(NTR):
                                r_ = np.random.default_rng(1000 * m + t + 7 * k + seed); Di = pf[r_.choice(len(pf), size=m, replace=m > len(pf))]
                                for an, al in algs.items(): acc[an].append(float([n for n, _ in al.fit_and_rank(D_obs, Di)][0] == tn))
                        for an in algs: res.setdefault(f"{proxy}|{graph}|x{gs}|{sname}|m{m}", {}).setdefault(an, []).append(float(np.mean(acc[an])))
                print(f"  seed {seed} {proxy}/{graph}/x{gs}: xong", flush=True)
    json.dump(res, open(f"{rd.OUT}/rca_ablation.json", "w"), indent=1)
    print("\nTop-1 (trung bình 5 nguyên nhân) ± sd qua 3 hạt giống; m=10; đoán bừa 0.07")
    algs = list(next(iter(res.values())))
    for scen in ("in", "shift"):
        print(f"\n[{ 'in-distribution' if scen == 'in' else 'DDE LỆCH'}]"); print("biến thể".ljust(28), *[a[:11].rjust(12) for a in algs])
        for key in [k for k in res if f"|{scen}|m10" in k]: print(key.split(f"|{scen}")[0].ljust(28), *[f"{np.mean(res[key][a]):.2f}±{np.std(res[key][a]):.2f}".rjust(12) for a in algs])


if __name__ == "__main__":
    main()
