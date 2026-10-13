"""Lõi ứng dụng: MỘT bộ sinh dùng chung cho hiển thị, huấn luyện và chẩn đoán (chế độ ổn định: độ cứng modal x10, KSCALE=10, TAG=_k10).
 Kênh được CHẨN ĐOÁN: gia tốc và micro = nền khỏe thật (NoiseGen) + g * (DDE lỗi - DDE khỏe), đúng như bộ dữ liệu huấn luyện.
 Kênh CHỈ HIỂN THỊ: dịch chuyển tương đối dụng cụ-phôi và lực (đầu ra thô của mô phỏng DDE + chuỗi cảm biến), không dùng để chẩn đoán."""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); PROTO = os.path.dirname(HERE); ROOT = os.path.dirname(PROTO); CS = os.path.join(ROOT, "cutting_sim")
os.environ.setdefault("KSCALE", "10"); os.environ["TAG"] = "_k10" if os.environ["KSCALE"] == "10" else os.environ.get("TAG", "")
for p in (CS, ROOT, PROTO): sys.path.insert(0, p)
os.chdir(CS)

import rca_data as rd  # noqa: E402

FAULTS = {  # nhãn -> (lớp mô phỏng, khoảng mức độ, trong mô hình?)
    "HEALTHY": ("NORMAL", (1.0, 1.0), True), "STIFFNESS": ("STIFFNESS", rd.SEV["STIFFNESS"], True), "DAMPING": ("DAMPING", rd.SEV["DAMPING"], True), "TOOL_WEAR": ("TOOL_WEAR", rd.SEV["TOOL_WEAR"], True),
    "DEPTH_OVERLOAD": ("DEPTH_OVERLOAD", rd.SEV["DEPTH_OVERLOAD"], True), "RPM_MISMATCH": ("RPM_MISMATCH", (0.03, 0.15), True), "IMBALANCE": ("IMBALANCE", (5.0, 50.0), False), "RUNOUT": ("RUNOUT", (2.0, 10.0), False)}
RPM_RANGE = (9000, 12000); AE_VALUES = [0.0635, 0.1905, 0.3175]; FT_MM = 0.0558; Z = 2
S = {}


def sev_from01(label, s01, sign=1):
    cls, (lo, hi), _ = FAULTS[label]
    if label == "HEALTHY": return 1.0
    s = float(np.clip(s01, 0, 1))
    if label in ("STIFFNESS", "DAMPING"): return hi - (hi - lo) * s            # hệ số nhân nhỏ = lỗi nặng: 100 % = nặng nhất (lo)
    v = lo + (hi - lo) * s; return 1.0 + sign * v if label == "RPM_MISMATCH" else v


def mrow_of(rpm, ae, up):
    return np.array([5, 0, 0, 0, 1 if up else 0, float(rpm), float(ae), FT_MM * Z * float(rpm) / 60.0, Z], float)


def _window(args):
    """Một cửa sổ 0.3 s: nền khỏe + g (lỗi - khỏe). Trả về 30 đặc trưng (+ dữ liệu hiển thị nếu show)."""
    mrow, cls, sev, seed, u, show = args; g = rd._G; rng = np.random.default_rng(seed)
    b = g["gen"].signals(mrow, u, rng, g["n_a"], g["n_m"]); (d0, sig0) = rd.dde_signals(mrow, "NORMAL", 1.0, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], None, True)
    if cls == "NORMAL": sig, sigF = b, sig0
    else:
        (d1, sigF) = rd.dde_signals(mrow, cls, sev, seed, g["fs_a"], g["sr"], g["n_a"], g["n_m"], None, True); n = min(len(d1[0]), len(b[0])), min(len(d1[2]), len(b[2]))
        sig = (b[0][:n[0]] + g["gacc"] * (d1[0][:n[0]] - d0[0][:n[0]]), b[1][:n[0]] + g["gacc"] * (d1[1][:n[0]] - d0[1][:n[0]]), b[2][:n[1]] + g["gmic"] * (d1[2][:n[1]] - d0[2][:n[1]]))
    feat = rd.rca_features(sig[0], sig[1], sig[2], g["fs_a"], g["sr"], mrow)
    if not show: return feat, None
    t = sigF["t"]; k = (t >= 0.50) & (t < 0.54)
    return feat, {"acc": sig[0], "acc_healthy": b[0], "mic": sig[2], "mic_healthy": b[2], "vib": sigF["vib_x"], "vib_healthy": sig0["vib_x"], "vib_y": sigF["vib_y"], "vib_y_healthy": sig0["vib_y"], "force": sigF["F_x"], "force_healthy": sig0["F_x"], "t_dde": t, "fs_a": g["fs_a"], "sr": g["sr"], "fs_dde": float(sigF["fs"])}


def init(workers=6, verbose=True):
    """Tải mô hình, khởi động numba, tạo pool (fork) sau khi khởi động để các tiến trình con thừa hưởng mã đã biên dịch."""
    if S: return S
    t0 = time.time()
    from kg import KnowledgeGraph
    from retrieval import Retriever
    d, meta, TH, fs_a, sr, n_a, n_m = rd.load(); gen = rd.NoiseGen().fit(meta, TH, fs_a, sr); g = json.load(open(f"{rd.OUT}/rca_gain_effect{rd.TAG}.json"))
    import cond24
    import experiment as ex
    from bn import CondBN
    res = json.load(open(os.path.join(PROTO, "report", f"cond24_results{rd.TAG}.json"))); mm = res["model"]; model = CondBN()
    model.k = 24; model.beta = np.array(mm["beta"]); model.Sigma = np.array(mm["Sigma"]); model.P = np.linalg.inv(model.Sigma); model.sd = np.sqrt(np.diag(model.Sigma)); model.T = mm["T"]; sd = np.array(mm["sd0"]); mask = None
    edges = {tuple(k.split("|")): e for k, e in res["edges_table"].items()}
    S.update(gen=gen, fs_a=fs_a, sr=sr, n_a=n_a, n_m=n_m, g=g, model=model, sd=sd, res=res, edges=edges, kg=KnowledgeGraph(), retriever=Retriever(), ex=ex, cond24=cond24)
    rd._init(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"]); _window((mrow_of(10000, 0.1905, True), "TOOL_WEAR", 1.3, 1, np.zeros(12), False))      # khởi động numba trong tiến trình chính
    S["pool"] = get_context("fork").Pool(workers, initializer=rd._init, initargs=(gen, fs_a, sr, n_a, n_m, g["g_acc"], g["g_mic"]))
    if verbose: print(f"live init {time.time() - t0:.1f}s | KSCALE={rd.KSCALE} TAG={rd.TAG} | gains {g['g_acc']:.2e}/{g['g_mic']:.3f} | T={model.T:.2f}", flush=True)
    return S


def make_windows(rpm, ae, up, label, s01=0.5, sign=1, m=10, seed=0, show=True):
    """m cửa sổ cùng một đường cắt (cùng u). Trả về (đặc trưng (m,30), dữ liệu hiển thị của cửa sổ 0)."""
    cls, _, _ = FAULTS[label]; sev = sev_from01(label, s01, sign); mrow = mrow_of(rpm, ae, up); rng = np.random.default_rng(seed)
    us = rng.normal(0, np.sqrt(S["gen"].between), (m, 12))          # offset nền riêng từng cửa sổ (như m cut cùng điều kiện): khớp giao thức hiệu chuẩn T (các cửa sổ lấy từ nhiều cut)
    jobs = [(mrow, cls, sev, 1000 * seed + i + 7, us[i], show and i == 0) for i in range(m)]; out = S["pool"].map(_window, jobs, chunksize=1)
    return np.array([o[0] for o in out]), out[0][1], sev


def build_evidence_groups(mm, D, case_id, consts, edges, kg, retriever, T):
    """Gói bằng chứng ở mức NHÓM băng tần (cộng dồn đóng góp các băng trong nhóm), cùng lược đồ với run_demo.build_evidence."""
    from kg import CAUSES
    from run_demo import LIMITS
    from retrieval import FEATURE_THEMES
    c24 = S["cond24"]; STATES = ["HEALTHY"] + CAUSES; m = len(D); idx, p = mm.posterior(D); order = np.argsort(-p); post = [{"cause": STATES[idx[i]], "prob": float(p[i]), "rank": r + 1} for r, i in enumerate(order)]
    top = post[0]["cause"]; ct = STATES.index(top); xbar = D.mean(0); gi = {g: [i for i, f in enumerate(c24.FEATS) if c24.GROUP[f] == g] for g in c24.GROUPS}; kge = kg.compile_edges(); paths = []
    if ct > 0:
        llr = mm.contributions(D, ct, ref=0); lg = {g: float(llr[ix].sum()) for g, ix in gi.items()}
        for g in sorted(lg, key=lambda x: -abs(lg[x]))[:5]:
            e = edges[(top, g)]; sp = kge.get((top, g), {}).get("sign", None); sh = float(xbar[gi[g]].mean()); intg = (top, g) in kge
            paths.append({"node": g, "observed_shift": sh, "llr": lg[g], "weight": e["weight"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"], "in_kg": intg, "sign_pred": 0 if sp is None else int(sp), "sign_agrees": bool(sp in (None, 0) or np.sign(sh) == sp),
                          "path_text": " || ".join(kg.explain_edge(top, g)) if intg else "not in the knowledge graph (data-suggested edge; expert review needed)", "tier": "ASSUMED"})
    d2 = float(m * (xbar - mm.mu[ct]) @ mm.P @ (xbar - mm.mu[ct])); res_ = (xbar - mm.mu[ct]) / np.array(consts["node_sd_ref"]); rg = {g: float(res_[ix].mean()) for g, ix in gi.items()}
    unexpl = [{"node": g, "z": rg[g]} for g in sorted(rg, key=lambda x: -abs(rg[x])) if abs(rg[g]) > 3][:5]; retrieved = []
    if retriever is not None and ct > 0:
        mech = [kg.entities[r["target"]]["label"] for r in kg.out[top] if r["relation"] == "leads_to"]; low = [kg.entities[top]["label"]] + mech + [kg.entities[p_["node"]]["label"] for p_ in paths[:3] if p_["node"] in kg.entities]
        high = sorted({t for p_ in paths[:3] for t in FEATURE_THEMES.get(p_["node"], [])} | {"vibration"})
        if d2 > consts["q95_nominal"]: low += ["spindle", "axis", "tool run-out"]; high += ["machine", "diagnosis", "runout"]
        retrieved = retriever.retrieve(low, high, top_n=4)
    return {"case_id": case_id, "variant": "condition-aware Gaussian Bayesian network on 24 spectral features", "m": m, "temperature": float(T), "posterior": post, "top": top, "node_shifts": {g: float(xbar[ix].mean()) for g, ix in gi.items()}, "kg_paths": paths,
            "misfit": {"d2": d2, "threshold": float(consts["q95_nominal"]), "flag": bool(d2 > consts["q95_nominal"]), "unexplained_nodes": unexpl}, "suggested_checks_from_kg": {c["cause"]: kg.checks_for(c["cause"]) for c in post[:2] if c["cause"] != "HEALTHY"}, "retrieved": retrieved, "limitations": LIMITS, "allowed_ids": list(kg.entities) + ["HEALTHY"] + c24.GROUPS}


def diagnose(F, rpm, ae, up, case_id="live", backend=None):
    """Đặc trưng (m,30) -> 24 đặc trưng tương đối theo nền khỏe của đúng điều kiện -> hậu nghiệm, kiểm tra ngoài mô hình, bằng chứng, truy hồi, giải thích đã kiểm tra."""
    from llm_reasoner import TemplateReasoner, reason
    mrow = mrow_of(rpm, ae, up); a = S["ex"].cond_design([mrow[5]], [mrow[6]], [mrow[7]], [mrow[4]])[0]; mod = S["model"]; D = F[:, :24] / S["sd"] - mod.baseline(a); mm = mod.at(a)
    ev = build_evidence_groups(mm, D, case_id, S["res"]["misfit"], S["edges"], S["kg"], S["retriever"], mod.T); out = reason(ev, backend or TemplateReasoner(), case_id=case_id); return D, ev, out
