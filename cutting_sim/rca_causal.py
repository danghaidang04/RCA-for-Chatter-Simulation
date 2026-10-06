"""RCA NHÂN QUẢ trên dữ liệu mô phỏng đã hiệu chỉnh nền (đúng giao thức của repo: D_obs bình thường + m mẫu lỗi D_int -> xếp hạng nút nguyên nhân gốc, Top-1/3/5).

 Nút (14): 5 nút NGUYÊN NHÂN GỐC = ước lượng tham số vật lý từ cảm biến ("cảm biến mềm": độ cứng, độ cản, lực cắt/mòn, độ sâu, rpm)
           + 9 nút CẢM BIẾN (log gia tốc, mức mic, 2 chỉ số chatter, 3 băng phổ gia tốc, 2 băng phổ mic).
 Đồ thị nhân quả (tri thức vật lý, ghi rõ trong báo cáo): mỗi nguyên nhân gốc -> các đại lượng cảm biến mà vật lý cho phép nó ảnh hưởng.
 Thuật toán: BRCD, RCD, RCG, SmoothTraversal, BARO, SimpleRCA (lớp trong run_full_paper_benchmark.py).
 Cảnh báo chống 'inverse crime': cảm biến mềm huấn luyện trên tham số DDE danh định; kịch bản 'lệch' kiểm trên tham số DDE khác.
"""
import json, os, sys
from multiprocessing import Pool
from types import SimpleNamespace
import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rca_data as rd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_full_paper_benchmark as rb  # noqa: E402

R = 3
FAULTS = rd.CLASSES[1:]
PROXY = ["Stiffness_est", "Damping_est", "ToolWear_est", "CutDepth_est", "SpindleRPM_est"]
SENS = ["log10_acc", "mic_dB", "chatter_acc", "chatter_mic", "acc_low", "acc_res", "acc_high", "mic_low", "mic_high"]
NODES = PROXY + SENS
EDGES = [("Stiffness_est", "acc_res"), ("Stiffness_est", "chatter_acc"), ("Stiffness_est", "log10_acc"),
         ("Damping_est", "acc_res"), ("Damping_est", "chatter_acc"), ("Damping_est", "chatter_mic"),
         ("ToolWear_est", "mic_dB"), ("ToolWear_est", "mic_low"), ("ToolWear_est", "log10_acc"), ("ToolWear_est", "acc_low"),
         ("CutDepth_est", "log10_acc"), ("CutDepth_est", "mic_dB"), ("CutDepth_est", "acc_low"), ("CutDepth_est", "acc_high"), ("CutDepth_est", "mic_high"),
         ("SpindleRPM_est", "acc_low"), ("SpindleRPM_est", "chatter_acc"), ("SpindleRPM_est", "chatter_mic"), ("SpindleRPM_est", "mic_low")]


def sens_nodes(F):
    """30 đặc trưng -> 9 nút cảm biến (băng phổ gộp)."""
    acc = F[:, 4:14]; mic = F[:, 14:24]
    return np.c_[F[:, 0], F[:, 1], F[:, 2], F[:, 3], acc[:, 0:3].mean(1), acc[:, 4:7].mean(1), acc[:, 7:10].mean(1), mic[:, 0:3].mean(1), mic[:, 6:10].mean(1)]


def make_system():
    dag = nx.DiGraph(); dag.add_nodes_from(range(len(NODES))); idx = {n: i for i, n in enumerate(NODES)}
    for u, v in EDGES: dag.add_edge(idx[u], idx[v])
    return SimpleNamespace(d=len(NODES), node_names=NODES, dag=dag, candidate_indices=list(range(len(NODES))), root_indices=[idx[p] for p in PROXY],
                           fault_types=FAULTS, fault_to_node=dict(zip(FAULTS, PROXY)), name_to_idx=idx)


def gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, mode, seed0):
    rng = np.random.default_rng(seed0); ucut = np.unique(cid); jobs = []
    for r in range(R):
        u_cut = {c: rng.normal(0, np.sqrt(gen.between)) for c in ucut}
        for i in range(len(meta)): jobs.append((meta[i], r, seed0 + 100000 * r + i, u_cut[cid[i]], rd.CLASSES, mode, "full"))
    with Pool(10, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"])) as pool:
        out = pool.map(rd.make_row_lat, jobs, chunksize=8)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])        # (R*n, 6, 30), (R*n, 6, 5)


def main():
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect.json"))
    cid = (meta[:, 0] * 1000 + meta[:, 1]).astype(int); ucut = np.unique(cid); n = len(meta)
    Fn, Ln = gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "nominal", 11); print("sinh nominal", Fn.shape, flush=True)
    Fs, Ls = gen_set(gen, meta, cid, fs_a, sr, n_a, n_m, g, "shift", 22); print("sinh shifted", Fs.shape, flush=True)
    rng = np.random.default_rng(0); by = {}; train_c, test_c = set(), set()
    for c in ucut: by.setdefault(meta[cid == c][0, 6], []).append(c)
    for ae, cs in by.items():
        cs = list(rng.permutation(cs)); nt = max(1, int(round(0.4 * len(cs)))) if len(cs) >= 3 else 0; test_c |= set(cs[:nt]); train_c |= set(cs[nt:])
    row_cut = np.tile(cid, R); mtr = np.isin(row_cut, list(train_c)); mte = np.isin(row_cut, list(test_c))
    print(f"đường cắt: train {len(train_c)}, test {len(test_c)}", flush=True)
    nf = Fn.shape[2]; flat = lambda A, m: A[m].reshape(-1, A.shape[-1])
    Xtr, Ytr = flat(Fn, mtr), flat(Ln, mtr); gtr = np.repeat(row_cut[mtr], 6)
    # cảm biến mềm: GB hồi quy log hệ số ẩn từ 30 đặc trưng; dự đoán chéo (cross-fit) cho train để D_obs không quá lạc quan
    # 4 nút đầu: cảm biến mềm (GB hồi quy log hệ số ẩn từ 30 đặc trưng). Nút rpm: ĐO TRỰC TIẾP từ bộ điều khiển (log(N_thực/N_kế hoạch) + nhiễu đọc 0.5%), như Spindle_Speed trong mtc.csv
    SOFT = range(4); rn = np.random.default_rng(5)
    soft = [HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Xtr, Ytr[:, j]) for j in SOFT]
    oof = np.zeros_like(Ytr)
    for a_, b_ in GroupKFold(5).split(Xtr, groups=gtr):
        for j in SOFT: oof[b_, j] = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Xtr[a_], Ytr[a_, j]).predict(Xtr[b_])
    oof[:, 4] = Ytr[:, 4] + rn.normal(0, 0.005, len(Ytr))
    rmse = np.sqrt(((oof - Ytr) ** 2).mean(0)); print("cảm biến mềm: RMSE ngoài mẫu của log-hệ-số:", dict(zip(PROXY, np.round(rmse, 3))), "| sd thật của nhân tố:", np.round(Ytr.std(0), 3), flush=True)
    to_nodes = lambda X, P: np.c_[P, sens_nodes(X)]
    obs_mask = np.tile(np.arange(6), mtr.sum()) == 0
    D_obs = to_nodes(Xtr, oof)[obs_mask]; MU, SD = D_obs.mean(0), D_obs.std(0) + 1e-9; zs = lambda D: (D - MU) / SD     # chuẩn hóa mọi nút theo D_obs (công bằng cho mọi thuật toán)
    D_obs = zs(D_obs); print("D_obs (bình thường, train):", D_obs.shape, flush=True)
    system = make_system()
    algs = {"BRCD": rb.BRCD(system), "RCD": rb.RCD(system), "RCG": rb.RCG(system), "SmoothTraversal": rb.SmoothTraversal(system), "BARO": rb.BARO(system), "SimpleRCA": rb.SimpleRCA(system)}
    scen = {"in-distribution (DDE danh định)": Fn, "tham số DDE LỆCH (kiểm tra bền)": Fs}
    ms = [5, 10, 20, 50]; n_trials = 100; results = {}
    for sname, FT in scen.items():
        Xte = flat(FT, mte); Yte = flat({'n': Ln, 's': Ls}['n' if sname.startswith('in') else 's'], mte); Pte = np.c_[[soft[j].predict(Xte) for j in SOFT]].T; Pte = np.c_[Pte, Yte[:, 4] + rn.normal(0, 0.005, len(Yte))]; Dte = zs(to_nodes(Xte, Pte)); cls = np.tile(np.arange(6), mte.sum())
        results[sname] = {a: {m: {f: [] for f in FAULTS} for m in ms} for a in algs}
        for m in ms:
            for k, f in enumerate(FAULTS):
                pool_f = Dte[cls == k + 1]; true_node = system.fault_to_node[f]
                for t in range(n_trials):
                    r_ = np.random.default_rng(1000 * m + t + 7 * k); D_int = pool_f[r_.choice(len(pool_f), size=m, replace=m > len(pool_f))]
                    for an, al in algs.items():
                        rk = [nm for nm, _ in al.fit_and_rank(D_obs, D_int)]
                        results[sname][an][m][f].append((rk[0] == true_node, true_node in rk[:3], true_node in rk[:5]))
        print(f"\n== {sname}: Top-1 trung bình trên 5 nguyên nhân (đoán bừa 1/14 = 0.07; Top-3 0.21; Top-5 0.36)")
        print("thuật toán".ljust(18), *[f"m={m}".rjust(18) for m in ms])
        for an in algs:
            row = []
            for m in ms:
                v = np.array([x for f in FAULTS for x in results[sname][an][m][f]], float).mean(0); row.append(f"{v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}".rjust(18))
            print(an.ljust(18), *row)
        print("  (định dạng: Top-1/Top-3/Top-5)")
    print("\nTop-1 theo nguyên nhân (m=10), in-distribution:")
    s0 = list(scen)[0]; print("thuật toán".ljust(18), *[f.rjust(15) for f in FAULTS])
    for an in algs: print(an.ljust(18), *[f"{np.mean([x[0] for x in results[s0][an][10][f]]):.2f}".rjust(15) for f in FAULTS])
    ser = {s: {a: {str(m): {f: [float(np.mean([x[i] for x in results[s][a][m][f]])) for i in range(3)] for f in FAULTS} for m in ms} for a in algs} for s in scen}
    json.dump({"soft_sensor_rmse": dict(zip(PROXY, rmse.tolist())), "results": ser, "edges": EDGES, "nodes": NODES}, open(f"{rd.OUT}/rca_causal.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for a_, s in zip(ax, scen):
        for an in algs: a_.plot(ms, [np.mean([x[0] for f in FAULTS for x in results[s][an][m][f]]) for m in ms], "o-", label=an)
        a_.set_title(s, fontsize=9); a_.set_xlabel("interventional samples m"); a_.axhline(1 / 14, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel("Top-1 (mean over 5 root causes)"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{rd.OUT}/rca_causal.png", dpi=130)


if __name__ == "__main__":
    main()
