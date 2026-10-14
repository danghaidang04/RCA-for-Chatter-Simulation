"""Phiên mô phỏng chạy LIÊN TỤC: mỗi 'tick' sinh một cửa sổ 0.3 s từ máy (khỏe, hoặc có lỗi tăng dần sau khi bấm Apply). Bộ phát hiện bất thường (CUSUM trên khoảng cách Mahalanobis của cửa sổ so với nền khỏe
của đúng điều kiện) chỉ dùng cảm biến; khi báo động, thu thêm 10 cửa sổ rồi chạy xếp hạng nguyên nhân gốc (mạng Bayes), dựng đồ thị nhân quả; lời giải thích gọi riêng sau."""
import json
import os
import time
import uuid

import numpy as np

from app import live

HERE = os.path.dirname(os.path.abspath(__file__)); SESS = {}; N_DIAG = 10; N_CAL = 10      # N_CAL: cửa sổ đầu sau Start dùng để học nền khỏe của chính máy/điều kiện này
DET_PATH = os.path.join(HERE, "report", "detector.json")


def detector_params():
    P = {"mu": 24.0, "sigma": 8.0, "k": 0.5, "h": 8.0, "k_l": 1.0, "h_l": 8.0}
    if os.path.exists(DET_PATH): P.update(json.load(open(DET_PATH))["params"])
    return P


def window_llr(dx, mm):
    """Tỉ số hợp lẽ lớn nhất của một cửa sổ cho 'có một trong 5 lỗi đã biết' so với 'khỏe' (dịch chuyển Gauss cùng Sigma): max_c [dx' P mu_c - 0.5 mu_c' P mu_c]."""
    Pm = mm.mu[1:] @ mm.P; return float(np.max(Pm @ dx - 0.5 * np.einsum("ck,ck->c", Pm, mm.mu[1:])))


def window_stat(feat, a):
    """24 đặc trưng tương đối theo nền khỏe hồi quy của điều kiện (dùng cho chẩn đoán)."""
    mod = live.S["model"]; return feat[:24] / live.S["sd"] - mod.baseline(a)


def dec(y, n):
    y = np.asarray(y, float); s = max(1, len(y) // n); return y[::s]


class Session:
    def __init__(s, rpm, ae, up, seed, backend, consent, par=None):
        s.par = par; s.pj = live.pj_of(par); s.sid = uuid.uuid4().hex[:12]; s.rpm, s.ae, s.up = rpm, ae, up; s.rng = np.random.default_rng(seed); s.seed = seed; s.tick = 0; s.fault = None; s.xs = []; s.cusum = 0.0; s.alarm_tick = None; s.diag = None; s.ev = None
        s.xref = None; s.cusum_l = 0.0; s.mm = live.S['model'].at(live.S['ex'].cond_design([live.mrow_of(rpm, ae, up)[5]], [live.mrow_of(rpm, ae, up)[6]], [live.mrow_of(rpm, ae, up)[7]], [live.mrow_of(rpm, ae, up)[4]])[0]); s.backend, s.consent = backend, consent; s.mrow = live.mrow_of(rpm, ae, up); s.a = live.S["ex"].cond_design([s.mrow[5]], [s.mrow[6]], [s.mrow[7]], [s.mrow[4]])[0]; s.created = time.time(); s.params = detector_params()

    def apply(s, label, sev01, ramp, sign=1):
        cls, _, in_model = live.FAULTS[label]; final = live.sev_from01(label, sev01, sign); s.fault = {"label": label, "cls": cls, "final": final, "t0": s.tick, "ramp": max(1, int(ramp)), "in_model": in_model, "sev01": sev01}
        s.cusum = 0.0; s.alarm_tick = None; s.diag = None; s.ev = None; s.xs = s.xs[-3:]

    def calibrated(s): return s.tick >= N_CAL

    def clear(s): s.fault = None; s.cusum = 0.0; s.alarm_tick = None; s.diag = None; s.ev = None; s.xs = []

    def step(s):
        f = s.fault; cls = "NORMAL"; val = 1.0; frac = 0.0
        if f: frac = float(np.clip((s.tick - f["t0"] + 1) / f["ramp"], 0, 1)); cls = f["cls"]; val = 1.0 + frac * (f["final"] - 1.0)
        u = s.rng.normal(0, np.sqrt(live.S["gen"].between), 12); feat, show = live._window((s.mrow, cls, val, 1000 * s.seed + s.tick + 11, u, True, s.pj)); x = window_stat(feat, s.a); P = s.params; s.xs.append(x); d2 = 0.0; z = 0.0
        fd_ = show["fs_dde"]; k0_ = int(0.50 * fd_); vx = show["vib"][k0_:]; vy = show["vib_y"][k0_:]; vr = float(np.sqrt(np.var(vx) + np.var(vy)) + 1e-9); s.vlog = getattr(s, "vlog", []) + [float(np.log(vr))]; zv = 0.0
        if s.tick < N_CAL: s.cal = getattr(s, "cal", []) + [x]
        if s.tick == N_CAL - 1: s.xref = np.mean(s.cal, axis=0); s.vmean = float(np.mean(s.vlog[:N_CAL])); s.vsd = float(max(np.std(s.vlog[:N_CAL]), P.get("v_floor", 0.05))); s.cusum_v = 0.0
        if s.xref is not None and s.tick >= N_CAL:
            dx = x - s.xref; d2 = float(dx @ live.S["model"].P @ dx); z = (d2 - P["mu"]) / P["sigma"]; s.cusum = max(0.0, s.cusum + z - P["k"]); llr = window_llr(dx, s.mm); s.cusum_l = max(0.0, s.cusum_l + llr - P["k_l"])
            zv = (s.vlog[-1] - s.vmean) / s.vsd; s.cusum_v = max(0.0, s.cusum_v + zv - P.get("k_v", 1.0)); mode = P.get("detector", "spectral")
            trig = {"vibration": s.cusum_v > P.get("h_v", 8.0), "spectral": (s.cusum > P["h"] or s.cusum_l > P["h_l"])}; trig["union"] = trig["vibration"] or trig["spectral"]
            if s.alarm_tick is None and trig[mode]: s.alarm_tick = s.tick
        llr_ = window_llr(x - s.xref, s.mm) if (s.xref is not None and s.tick >= N_CAL) else 0.0
        out = {"llr": llr_, "calibrating": s.tick < N_CAL, "tick": s.tick, "fault": (f["label"] if f else None), "fault_frac": frac, "d2": d2, "z": z, "vib_rms": vr, "zv": zv, "cusum_vib": getattr(s, "cusum_v", 0.0) / P.get("h_v", 8.0), "score": (getattr(s, "cusum_v", 0.0) / P.get("h_v", 8.0) if P.get("detector") == "vibration" else (max(s.cusum / P["h"], s.cusum_l / P["h_l"]) if P.get("detector", "spectral") == "spectral" else max(getattr(s, "cusum_v", 0.0) / P.get("h_v", 8.0), s.cusum / P["h"], s.cusum_l / P["h_l"]))), "cusum": 0.0, "cusum_generic": s.cusum / P["h"], "cusum_model": s.cusum_l / P["h_l"], "h": 1.0, "alarm": s.alarm_tick is not None, "alarm_tick": s.alarm_tick, "collecting": None, "diagnosis": None}
        out["cusum"] = out["score"]
        fa, sr, fd = show["fs_a"], show["sr"], show["fs_dde"]; k0 = int(0.50 * fd); n100 = int(0.1 * fd)
        out["signals"] = {"vib": {"y": dec(show["vib"][k0:], 240).round(3).tolist(), "dur_ms": 100, "unit": "µm"}, "acc": {"y": dec(show["acc"], 300).round(4).tolist(), "dur_ms": 300, "unit": "g"},
                          "mic": {"y": dec(show["mic"], 300).round(4).tolist(), "dur_ms": 300, "unit": "Pa"}, "force": {"y": dec(show["force"][k0:], 240).round(2).tolist(), "dur_ms": 100, "unit": "N"}}
        out["tip"] = {"x": dec(show["vib"][k0:k0 + n100], 320).round(3).tolist(), "y": dec(show["vib_y"][k0:k0 + n100], 320).round(3).tolist(), "dur_s": 0.1}
        if s.alarm_tick is not None and s.diag is None:
            out["collecting"] = min(N_DIAG, s.tick - s.alarm_tick)
            if s.tick - s.alarm_tick >= N_DIAG: s.diag = s._diagnose(); out["diagnosis"] = s.diag
        s.tick += 1; return out

    def _diagnose(s):
        from kg import CAUSES  # noqa: F401
        mod = live.S["model"]; mm = mod.at(s.a); D = np.array(s.xs[-N_DIAG:]); ev = live.build_evidence_groups(mm, D, s.sid, live.S["res"]["misfit"], live.S["edges"], live.S["kg"], live.S["retriever"], mod.T); s.ev = ev
        return {"posterior": ev["posterior"], "top": ev["top"], "misfit": ev["misfit"], "kg_paths": ev["kg_paths"], "retrieved": ev["retrieved"], "graph": graph_spec(ev), "injected": (s.fault["label"] if s.fault else None), "injected_in_model": (s.fault["in_model"] if s.fault else None)}

    def explain(s):
        from llm_reasoner import OpenRouterReasoner, TemplateReasoner, reason
        if s.ev is None: return None
        backend = TemplateReasoner()
        if s.backend == "openrouter":
            if not s.consent: raise PermissionError("consent not given")
            backend = OpenRouterReasoner(allow_external=True)
        out = reason(s.ev, backend, case_id=s.sid); return {"explanation": out["explanation"], "backend": out["backend"], "validated": out["validated"], "llm_violations": out["llm_violations"], "sources": s.ev.get("retrieved", [])}


def graph_spec(ev):
    """Đồ thị nhân quả của ca này: nguyên nhân -> cơ chế -> nhóm băng tần (cạnh KG) + cạnh gợi ý từ dữ liệu cho nguyên nhân hạng nhất; nút đặc trưng tô theo dịch chuyển quan sát."""
    kg = live.S["kg"]; edges = live.S["edges"]; c24 = live.S["cond24"]; from kg import CAUSES
    prob = {p["cause"]: p["prob"] for p in ev["posterior"]}; top = ev["top"]; nodes = []; E = []
    for c in CAUSES: nodes.append({"id": c, "type": "cause", "label": kg.entities[c]["label"], "value": prob.get(c, 0.0), "top": c == top})
    mech_of = {r["source"]: r["target"] for r in kg.relations if r["relation"] == "leads_to"}
    for c, m in mech_of.items(): nodes.append({"id": m, "type": "mechanism", "label": kg.entities[m]["label"]}); E.append({"src": c, "dst": m, "kind": "kg", "weight": 0.0, "on_top": c == top})
    unex = {u["node"] for u in ev["misfit"]["unexplained_nodes"]}
    for g in c24.GROUPS: nodes.append({"id": g, "type": "feature", "label": (kg.entities[g]["label"] if g in kg.entities else g), "value": ev["node_shifts"].get(g, 0.0), "unexplained": g in unex})
    for r in kg.relations:
        if r["relation"] == "manifests_as" and r["target"] in c24.GROUPS:
            cause = next((c for c, m in mech_of.items() if m == r["source"]), None); w = edges.get((cause, r["target"]), {}).get("weight", 0.0); E.append({"src": r["source"], "dst": r["target"], "kind": "kg", "weight": w, "on_top": cause == top, "sign": r["sign"]})
    kgpairs = {(next((c for c, m in mech_of.items() if m == r["source"]), None), r["target"]) for r in kg.relations if r["relation"] == "manifests_as"}
    if top in CAUSES:
        for g in c24.GROUPS:
            e = edges[(top, g)]
            if (top, g) not in kgpairs and (e["ci_lo"] > 0 or e["ci_hi"] < 0) and abs(e["weight"]) >= 0.5: E.append({"src": top, "dst": g, "kind": "data", "weight": e["weight"], "on_top": True})
    return {"nodes": nodes, "edges": E, "top": top}
