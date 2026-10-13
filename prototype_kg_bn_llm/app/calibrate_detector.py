"""Hiệu chuẩn bộ phát hiện bất thường. Ghi chuỗi (d², LLR, log RMS dao động tương đối) của nhiều phiên khỏe và phiên có lỗi tăng dần rồi chọn tham số CUSUM OFFLINE cho ba phương án:
 'vibration' (chỉ dao động tương đối dao-phôi, so với nền khỏe của chính máy trong 10 cửa sổ đầu), 'spectral' (d² + tỉ số hợp lẽ trên gia tốc/micro), 'union' (một trong hai).
Ngưỡng chọn sao cho 0 báo động nhầm trên 40 phiên khỏe dùng để chỉnh; kiểm lại trên 20 phiên khỏe còn lại (hold-out). Chạy: python -m app.calibrate_detector"""
import json
import os
import time

import numpy as np

from app import live, stream

HERE = os.path.dirname(os.path.abspath(__file__)); CONDS = [(9000, 0.0635, False), (10500, 0.1905, True), (12000, 0.3175, False), (9500, 0.3175, True), (11500, 0.0635, False)]; T_APPLY = 14; N = 55; NC = stream.N_CAL


def session(cond, seed, fault=None, sev=0.5, ramp=6, sign=1):
    s = stream.Session(*cond, seed, "template", False); d = []
    for t in range(N):
        if fault and t == T_APPLY: s.apply(fault, sev, ramp, sign)
        o = s.step(); d.append((o["d2"], o["llr"], s.vlog[-1]))
    return np.array(d)


def alarm_vib(seq, k, h, floor):
    v = seq[:, 2]; m = v[:NC].mean(); sd = max(v[:NC].std(), floor); S = 0.0
    for i in range(NC, len(v)):
        S = max(0.0, S + (v[i] - m) / sd - k)
        if S > h: return i
    return None


def alarm_spec(seq, mu, sigma, k, h, k_l, h_l):
    S = L = 0.0
    for i in range(NC, len(seq)):
        S = max(0.0, S + (seq[i, 0] - mu) / sigma - k); L = max(0.0, L + seq[i, 1] - k_l)
        if S > h or L > h_l: return i
    return None


def main():
    live.init(workers=2); t0 = time.time(); H = [session(CONDS[i % 5], 900 + i) for i in range(60)]; print("healthy done", round(time.time() - t0), "s", flush=True); F = {}
    for f in live.FAULTS:
        if f == "HEALTHY": continue
        for sev in (0.5, 0.9): F[(f, sev)] = [session(CONDS[i % 5], 1200 + 31 * i, f, sev, 6, 1 if i % 2 == 0 else -1) for i in range(12)]
    print("faults done", round(time.time() - t0), "s", flush=True); tune, hold = H[:40], H[40:]; allf = [d for v in F.values() for d in v]; mu = float(np.mean([h[NC:NC + 20, 0].mean() for h in H])); sigma = float(np.mean([h[NC:NC + 20, 0].std() for h in H]))
    def best(fn, grid, name):
        b = None
        for p in grid:
            if sum(fn(d, *p) is not None for d in tune) == 0:
                det = np.mean([fn(d, *p) is not None for d in allf])
                if b is None or det > b[0]: b = (det, p)
        print(name, "best params", b[1] if b else None, "detection over all fault sessions", round(b[0], 3) if b else None, flush=True); return b[1]
    pv = best(alarm_vib, [(k, h, fl) for k in (0.5, 1.0, 2.0) for h in np.arange(2, 60, 2.0) for fl in (0.02, 0.05, 0.1)], "vibration")
    pg = best(lambda d, k, h, kl, hl: alarm_spec(d, mu, sigma, k, h, kl, hl), [(k, h, kl, hl) for k in (0.5, 1.0, 2.0) for h in np.arange(10, 100, 10.0) for kl in (0.5, 1.0, 2.0) for hl in np.arange(2, 40, 2.0)], "spectral")
    P = {"mu": mu, "sigma": sigma, "k": float(pg[0]), "h": float(pg[1]), "k_l": float(pg[2]), "h_l": float(pg[3]), "k_v": float(pv[0]), "h_v": float(pv[1]), "v_floor": float(pv[2])}
    modes = {"vibration": lambda d: alarm_vib(d, pv[0], pv[1], pv[2]), "spectral": lambda d: alarm_spec(d, mu, sigma, *pg)}; modes["union"] = lambda d: (lambda a, b: (min(x for x in (a, b) if x is not None) if (a is not None or b is not None) else None))(modes["vibration"](d), modes["spectral"](d))
    res = {"params": P, "n_healthy_sessions": len(H), "ticks_per_session": N, "n_tune": len(tune), "n_holdout": len(hold), "modes": {}}
    for mode, fn in modes.items():
        r = {"healthy_false_alarm_tune": float(np.mean([fn(d) is not None for d in tune])), "healthy_false_alarm_holdout": float(np.mean([fn(d) is not None for d in hold])), "faults": {}}
        for (f, sev), v in F.items():
            al = [fn(d) for d in v]; det = [a for a in al if a is not None and a >= T_APPLY]; early = [a for a in al if a is not None and a < T_APPLY]
            r["faults"][f"{f}@{sev}"] = {"detected": len(det) / len(al), "false_before_onset": len(early) / len(al), "median_delay_ticks": (float(np.median([a - T_APPLY for a in det])) if det else None)}
        res["modes"][mode] = r; print(mode, "false alarm tune/holdout", r["healthy_false_alarm_tune"], r["healthy_false_alarm_holdout"], "| mean detection", round(float(np.mean([x["detected"] for x in r["faults"].values()])), 3), flush=True)
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(HERE, "report", "detector_v3.json"), "w"), indent=1)
    for mode in res["modes"]:
        print("==", mode); [print("  ", k, v) for k, v in res["modes"][mode]["faults"].items()]
    live.S["pool"].terminate()


if __name__ == "__main__":
    main()
