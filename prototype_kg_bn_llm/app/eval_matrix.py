"""Ma trận kiểm tra tự động: mọi lỗi x 3 điều kiện x 3 mức độ x hạt giống, qua ĐÚNG hàm của ứng dụng (live.make_windows + live.diagnose). Kết quả TRONG PHÂN PHỐI (cùng bộ sinh huấn luyện).
Chạy: python -m app.eval_matrix   -> app/report/eval_matrix.json"""
import json
import os
import time

import numpy as np

from app import live

HERE = os.path.dirname(os.path.abspath(__file__)); COND = [(9000, 0.0635, False), (10500, 0.1905, True), (12000, 0.3175, False)]; SEVS = [0.2, 0.5, 0.8]; N_SEED = 5


def main():
    live.init(workers=6); causes = ["STIFFNESS", "DAMPING", "TOOL_WEAR", "DEPTH_OVERLOAD", "RPM_MISMATCH"]; rows = []; t0 = time.time()
    for fault in live.FAULTS:
        sevs = [0.5] if fault == "HEALTHY" else SEVS
        for ci, (rpm, ae, up) in enumerate(COND):
            for s in sevs:
                for seed in range(N_SEED if fault != "HEALTHY" else 3 * N_SEED):
                    F, _, _ = live.make_windows(rpm, ae, up, fault, s, 1 if seed % 2 == 0 else -1, m=10, seed=100 * ci + seed + (7 if fault == "HEALTHY" else 0), show=False); nodes, ev, out = live.diagnose(F, rpm, ae, up)
                    rows.append({"fault": fault, "cond": ci, "sev": s, "top": ev["posterior"][0]["cause"], "p_top": ev["posterior"][0]["prob"], "flag": bool(ev["misfit"]["flag"]), "d2": ev["misfit"]["d2"], "validated": out["validated"]})
    r = np.array([(x["fault"], x["top"]) for x in rows]); fl = lambda f: float(np.mean([x["flag"] for x in rows if x["fault"] == f]))
    mod = [x for x in rows if x["fault"] in causes]; hl = [x for x in rows if x["fault"] == "HEALTHY"]; conf = {f: {} for f in live.FAULTS}
    for x in rows: conf[x["fault"]][x["top"]] = conf[x["fault"]].get(x["top"], 0) + 1
    res = {"n_runs": len(rows), "seconds": round(time.time() - t0, 1), "top1_modelled": float(np.mean([x["fault"] == x["top"] for x in mod])), "top1_per_cause": {c: float(np.mean([x["top"] == c for x in rows if x["fault"] == c])) for c in causes},
           "healthy_flag_rate": fl("HEALTHY"), "healthy_false_alarm": float(np.mean([x["top"] != "HEALTHY" for x in hl])), "modelled_flag_rate": float(np.mean([x["flag"] for x in mod])), "imbalance_flag_rate": fl("IMBALANCE"), "runout_flag_rate": fl("RUNOUT"),
           "mean_top_prob_when_wrong": float(np.mean([x["p_top"] for x in mod if x["fault"] != x["top"]])), "all_explanations_validated": bool(all(x["validated"] for x in rows)), "confusion": conf, "note": "in-distribution simulator results (same generator for training and testing); 3 conditions x 3 severities x 5 seeds"}
    os.makedirs(os.path.join(HERE, "report"), exist_ok=True); json.dump(res, open(os.path.join(HERE, "report", "eval_matrix.json"), "w"), indent=1); json.dump(rows, open(os.path.join(HERE, "report", "eval_matrix_rows.json"), "w"))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "confusion"}); [print(" ", f, conf[f]) for f in conf]
    live.S["pool"].terminate()


if __name__ == "__main__":
    main()
