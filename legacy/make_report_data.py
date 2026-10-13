"""Thu thập số liệu cho báo cáo (Vi/En) bằng cách chạy lại mã gốc của dự án.

Tạo:
  report_results.json    - benchmark RCA 14 biến (Top-1/3/5) + hội tụ xác suất hậu nghiệm THÔ của BRCD
  physics_results.json   - 6 kịch bản mô phỏng DDE + mạng Bayes + Monte-Carlo 120 lần
  physics_waveforms.npz  - dạng sóng/phổ của 3 kịch bản dùng để vẽ hình

Chạy: python make_report_data.py   (khoảng 3-4 phút)
"""
import json

import numpy as np

import rca_chatter_bayesian as rb_phys
import run_full_paper_benchmark as rb

SAMPLE_SIZES = [5, 10, 20, 50, 100]
N_SEEDS = 100
N_STREAM = 60
MAX_STEPS = 40


def benchmark_and_convergence():
    system = rb.CNCChatterTelemetryGraph()
    algos = {
        "BRCD": rb.BRCD(system), "RCD": rb.RCD(system), "RCG": rb.RCG(system),
        "SmoothTraversal": rb.SmoothTraversal(system), "BARO": rb.BARO(system),
        "SimpleRCA": rb.SimpleRCA(system),
    }
    D_obs = system.sample_observational_data(n_samples=5000, seed=12345)
    res = {a: {str(m): {f: {"top1": [], "top3": [], "top5": []} for f in system.fault_types}
               for m in SAMPLE_SIZES} for a in algos}
    for m in SAMPLE_SIZES:
        for f in system.fault_types:
            true_node = system.fault_to_node[f]
            for seed in range(N_SEEDS):
                D_int = system.sample_interventional_data(n_samples=m, fault_type=f, seed=1000 * m + seed)
                for name, model in algos.items():
                    ranked = [n for n, _ in model.fit_and_rank(D_obs, D_int)]
                    for k in (1, 3, 5):
                        res[name][str(m)][f][f"top{k}"].append(float(true_node in ranked[:k]))
    out = {"benchmark": {}}
    for a in res:
        out["benchmark"][a] = {}
        for m in res[a]:
            out["benchmark"][a][m] = {"by_fault": {}}
            for k in ("top1", "top3", "top5"):
                allv = np.concatenate([res[a][m][f][k] for f in system.fault_types])
                out["benchmark"][a][m][k] = [float(allv.mean()), float(allv.std() / np.sqrt(len(allv)))]
            for f in system.fault_types:
                v = res[a][m][f]["top1"]
                out["benchmark"][a][m]["by_fault"][f] = [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]

    # Hội tụ: xác suất hậu nghiệm THÔ của BRCD (không pha trộn với prior, không cộng nhiễu)
    brcd = algos["BRCD"]
    n_f = len(system.fault_types)
    conv = {"max_entropy": float(np.log(n_f)), "prior": 1.0 / n_f, "faults": {}}
    for f in system.fault_types:
        true_node = system.fault_to_node[f]
        P = np.zeros((N_STREAM, MAX_STEPS))
        H = np.zeros((N_STREAM, MAX_STEPS))
        for tr in range(N_STREAM):
            D_full = system.sample_interventional_data(n_samples=MAX_STEPS, fault_type=f, seed=30000 + tr * 7)
            for t in range(1, MAX_STEPS + 1):
                post, _ = brcd.get_posteriors(D_obs, D_full[:t])
                vec = np.array([post[system.fault_to_node[x]] for x in system.fault_types])
                vec = vec / vec.sum()
                P[tr, t - 1] = post[true_node]
                H[tr, t - 1] = -np.sum(vec * np.log(np.maximum(vec, 1e-12)))
        conv["faults"][f] = {"p_mean": P.mean(0).tolist(), "p_lo": np.percentile(P, 25, 0).tolist(),
                             "p_hi": np.percentile(P, 75, 0).tolist(), "h_mean": H.mean(0).tolist(),
                             "h_lo": np.percentile(H, 25, 0).tolist(), "h_hi": np.percentile(H, 75, 0).tolist()}
    out["convergence"] = conv
    return out


def physics():
    nominal = dict(fn=250.0, zeta=0.012, k=2.26e8, Kf=1.0e9, N_rpm=3000.0, a=5.0)
    scen = [
        ("Normal", nominal, "Normal"),
        ("Stiffness", dict(nominal, fn=250.0 * np.sqrt(0.5), k=nominal["k"] * 0.5, a=10.0), "Stiffness_Degradation"),
        ("Damping", dict(nominal, zeta=nominal["zeta"] * 0.15, N_rpm=3800.0, a=8.0), "Damping_Loss"),
        ("ToolWear", dict(nominal, Kf=nominal["Kf"] * 2.2), "Tool_Wear"),
        ("Depth", dict(nominal, N_rpm=3800.0, a=28.0), "Excessive_Cut_Depth"),
        ("RPM", dict(nominal, N_rpm=8800.0, a=12.0), "Unstable_Spindle_RPM"),
    ]
    bn = rb_phys.BayesianNetworkRCA()

    def observe(feats, p):
        return {
            "High_Vibration": bool(feats["rms_vib"] > 5.0),
            "Chatter_Regenerative_Peak": bool(feats["chatter_ratio"] > 0.45 and feats["rms_vib"] > 4.0),
            "Resonance_Frequency_Shift": bool(abs(feats["dom_freq"] - 250.0) > 30.0 and feats["rms_vib"] > 3.0),
            "High_Cutting_Force": bool(feats["mean_force"] > 450.0 or feats["peak_force"] > 1000.0),
            "Programmed_Depth_High": bool(p["a"] > 20.0),
            "RPM_In_Lobe_Valley": bool(p["N_rpm"] > 7000.0 or (4000 < p["N_rpm"] < 6000 and p["a"] > 10.0)),
        }

    rows, wave = [], {}
    for key, p, truth in scen:
        sim = rb_phys.DynamicChatterSimulator(fn=p["fn"], zeta=p["zeta"], k=p["k"], Kf=p["Kf"],
                                              N_rpm=p["N_rpm"], a=p["a"])
        t, y_um, force, dt = sim.simulate(t_end=0.25, dt=2e-6)
        feats = rb_phys.extract_features(t, y_um, force, dt, 250.0, p["N_rpm"])
        post = bn.compute_posterior(observe(feats, p))
        top = next(iter(post))
        rows.append({"key": key, "truth": truth, "params": {k: float(v) for k, v in p.items()},
                     "rms": float(feats["rms_vib"]), "dom_freq": float(feats["dom_freq"]),
                     "mean_force": float(feats["mean_force"]), "peak_force": float(feats["peak_force"]),
                     "diagnosed": top, "prob": float(post[top]), "correct": top == truth})
        if key in ("Normal", "Stiffness", "Depth"):
            sel = (t >= 0.12) & (t <= 0.20)
            wave[f"{key}_t"] = (t[sel] * 1000).astype(np.float32)
            wave[f"{key}_y"] = y_um[sel].astype(np.float32)
            wave[f"{key}_xf"] = np.asarray(feats["xf"], dtype=np.float32)
            wave[f"{key}_mag"] = np.asarray(feats["mag"], dtype=np.float32)

    np.random.seed(2026)
    hit1 = hit3 = total = 0
    for key, p, truth in scen:
        for trial in range(20):
            jk = p["k"] * np.random.uniform(0.95, 1.05)
            jz = p["zeta"] * np.random.uniform(0.9, 1.1)
            jf = p["Kf"] * np.random.uniform(0.95, 1.05)
            jfn = np.sqrt(jk / (p["k"] / ((2 * np.pi * p["fn"]) ** 2))) / (2 * np.pi)
            sim = rb_phys.DynamicChatterSimulator(fn=jfn, zeta=jz, k=jk, Kf=jf, N_rpm=p["N_rpm"], a=p["a"])
            t, y_um, force, dt = sim.simulate(t_end=0.15, dt=2.5e-6, seed=trial)
            feats = rb_phys.extract_features(t, y_um, force, dt, 250.0, p["N_rpm"])
            ranked = list(bn.compute_posterior(observe(feats, p)))
            hit1 += ranked[0] == truth
            hit3 += truth in ranked[:3]
            total += 1
    return ({"scenarios": rows, "mc": {"n": total, "top1": hit1 / total, "top3": hit3 / total}}, wave)


def main():
    out = benchmark_and_convergence()
    json.dump(out, open("report_results.json", "w"), indent=1)
    print("saved report_results.json")
    phys, wave = physics()
    json.dump(phys, open("physics_results.json", "w"), indent=1)
    np.savez_compressed("physics_waveforms.npz", **wave)
    print("saved physics_results.json, physics_waveforms.npz")
    print({k: (v["diagnosed"], round(v["prob"], 3), v["correct"]) for k, v in
           zip([s["key"] for s in phys["scenarios"]], phys["scenarios"])}, phys["mc"])


if __name__ == "__main__":
    main()
