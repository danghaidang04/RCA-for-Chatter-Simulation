"""Kiểm thử luồng của giao diện qua HTTP (máy chủ phải đang chạy): Start -> học nền -> Apply -> báo động -> chẩn đoán -> giải thích. python -m app.flow_test [FAULT] [severity] [backend]"""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, body=None, method=None):
    req = urllib.request.Request(BASE + path, data=(json.dumps(body).encode() if body is not None else None), headers={"Content-Type": "application/json"}, method=method or ("POST" if body is not None else "GET"))
    return json.load(urllib.request.urlopen(req, timeout=180))


def main():
    fault = sys.argv[1] if len(sys.argv) > 1 else "TOOL_WEAR"; sev = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9; backend = sys.argv[3] if len(sys.argv) > 3 else "template"
    j = call("/api/sim/start", {"rpm": 10500, "ae": 0.1905, "up": False, "seed": 4, "backend": backend, "consent": backend == "openrouter"}); sid = j["sid"]; applied = None; alarm = None; diag = None; t0 = time.time()
    for t in range(80):
        if t == 14: print("apply ->", call(f"/api/sim/{sid}/apply", {"fault": fault, "severity": sev, "ramp": 6, "sign": 1})); applied = t
        o = call(f"/api/sim/{sid}/step")
        if t in (0, 9, 10, 13): print(f"t={t:2d} calibrating={o['calibrating']} score={o['cusum']:.2f} alarm={o['alarm']}")
        if o["alarm"] and alarm is None: alarm = t; print(f"ALARM at window {t} (onset at {applied}, delay {t - applied}); score {o['cusum']:.2f} generic {o['cusum_generic']:.2f} model {o['cusum_model']:.2f}")
        if o["diagnosis"]: diag = o["diagnosis"]; print(f"diagnosis at window {t}"); break
    if diag is None: print("no alarm/diagnosis within 80 windows"); return
    print("ranking:", [(p["cause"], round(p["prob"], 2)) for p in diag["posterior"][:3]], "| misfit flag", diag["misfit"]["flag"], "| injected", diag["injected"])
    g = diag["graph"]; print("graph:", len(g["nodes"]), "nodes", len(g["edges"]), "edges; top", g["top"], "| data edges", sum(e["kind"] == "data" for e in g["edges"]))
    ex = call(f"/api/sim/{sid}/explain", method="POST"); print("explanation backend:", ex["backend"], "validated:", ex["validated"], "violations:", ex["llm_violations"]); print("summary:", ex["explanation"]["summary"][:300])
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
