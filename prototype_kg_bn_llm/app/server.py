"""Ứng dụng web: mô phỏng máy phay theo khối (Python) -> bốn loại tín hiệu -> chẩn đoán nguyên nhân gốc (mạng Bayes + KG) -> giải thích (mẫu hoặc LLM qua OpenRouter, có đồng ý).
Chạy: cd prototype_kg_bn_llm && python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
Khóa OpenRouter chỉ được đọc ở phía máy chủ từ .env/biến môi trường; không bao giờ gửi tới trình duyệt."""
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import live, stream

HERE = os.path.dirname(os.path.abspath(__file__)); PROTO = os.path.dirname(HERE)


@asynccontextmanager
async def lifespan(app):
    live.init(workers=int(os.environ.get("APP_WORKERS", "6"))); yield
    if live.S.get("pool"): live.S["pool"].terminate()


app = FastAPI(title="Milling simulator + RCA demo", lifespan=lifespan)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=[o for o in os.environ.get("ALLOWED_ORIGINS", "https://danghaidang04.github.io").split(",") if o], allow_methods=["*"], allow_headers=["*"])


class Req(BaseModel):
    rpm: float = Field(10500, ge=9000, le=12000)
    ae: float = Field(0.1905)
    up: bool = False
    fault: str = "HEALTHY"
    severity: float = Field(0.5, ge=0.0, le=1.0)
    rpm_sign: int = 1
    seed: int = 0
    backend: str = "template"          # template | openrouter
    consent: bool = False              # bắt buộc nếu backend = openrouter


def spectrum(x, fs, fmax=5000.0, n=300):
    x = np.asarray(x, float); x = x - x.mean(); P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 / len(x); f = np.fft.rfftfreq(len(x), 1 / fs); k = f <= fmax; f, P = f[k], P[k]
    edges = np.linspace(0, len(f), n + 1).astype(int); fb = np.array([f[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]); pb = np.array([P[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]); return fb, 10 * np.log10(pb + 1e-30)


def trace(y, y0, fs, unit="", use="", name="", maxn=3200):
    """Toàn bộ đoạn tín hiệu (giảm mẫu để vẽ) cho màn hình cuộn theo thời gian + phổ."""
    y = np.asarray(y, float); y0 = np.asarray(y0, float); step = max(1, len(y) // maxn); t = (np.arange(0, len(y), step) / fs) * 1e3; fy, dy = spectrum(y, fs); _, d0 = spectrum(y0, fs)
    return {"name": name, "unit": unit, "use": use, "fs": fs, "t_ms": t.round(4).tolist(), "y": y[::step].round(5).tolist(), "y_healthy": y0[::step].round(5).tolist(), "f_hz": fy.round(1).tolist(), "psd_db": dy.round(2).tolist(), "psd_db_healthy": d0.round(2).tolist()}


@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/info")
def info():
    S = live.S; env = __import__("llm_reasoner").load_env(); ev = None; p = os.path.join(HERE, "report", "eval_matrix.json")
    if os.path.exists(p): ev = json.load(open(p))
    return {"faults": {k: {"in_model": v[2], "range": v[1]} for k, v in live.FAULTS.items()}, "rpm_range": live.RPM_RANGE, "ae_values": live.AE_VALUES, "temperature": S["model"].T, "model_top1": S["res"]["nominal"]["top1"], "model_top1_shifted": S["res"]["shifted"]["top1"], "kscale": float(os.environ.get("KSCALE", "10")),
            "openrouter_key_present": bool(env.get("OPENROUTER_API_KEY")), "openrouter_model": env.get("OPENROUTER_MODEL") or "google/gemma-3-4b-it", "eval_matrix": ev}


@app.post("/api/simulate")
def simulate(r: Req):
    if r.fault not in live.FAULTS: raise HTTPException(400, "unknown fault")
    if min(abs(r.ae - a) for a in live.AE_VALUES) > 1e-6: raise HTTPException(400, f"ae must be one of {live.AE_VALUES} (trained range)")
    from llm_reasoner import OpenRouterReasoner, TemplateReasoner
    backend = TemplateReasoner()
    if r.backend == "openrouter":
        if not r.consent: raise HTTPException(400, "consent is required to send the evidence bundle to OpenRouter")
        backend = OpenRouterReasoner(allow_external=True)
    t0 = time.time(); F, show, sev = live.make_windows(r.rpm, r.ae, r.up, r.fault, r.severity, r.rpm_sign, m=10, seed=r.seed); t1 = time.time()
    nodes, ev, out = live.diagnose(F, r.rpm, r.ae, r.up, case_id="live", backend=backend); t2 = time.time(); fa, sr, fd = show["fs_a"], show["sr"], show["fs_dde"]; tdde = show["t_dde"]; k0 = int(0.50 * fd)
    sig = {"vib": trace(show["vib"][k0:], show["vib_healthy"][k0:], fd, unit="µm", use="display only (simulator output, not used for diagnosis)", name="Relative tool-workpiece vibration"),
           "acc": trace(show["acc"], show["acc_healthy"], fa, unit="g", use="diagnosed (baseline + g·Δ)", name="Accelerometer"), "mic": trace(show["mic"], show["mic_healthy"], sr, unit="Pa", use="diagnosed (baseline + g·Δ)", name="Microphone"),
           "force": trace(show["force"][k0:], show["force_healthy"][k0:], fd, unit="N", use="display only (simulator output, not used for diagnosis)", name="Force dynamometer")}
    n_tip = int(0.1 * fd); tip = {"fs": fd, "x": np.asarray(show["vib"][k0:k0 + n_tip]).round(4).tolist(), "y": np.asarray(show["vib_y"][k0:k0 + n_tip]).round(4).tolist(), "x_healthy": np.asarray(show["vib_healthy"][k0:k0 + n_tip]).round(4).tolist(), "y_healthy": np.asarray(show["vib_y_healthy"][k0:k0 + n_tip]).round(4).tolist()}
    cause, _, in_model = live.FAULTS[r.fault]
    return JSONResponse({"condition": {"rpm": r.rpm, "ae": r.ae, "up": r.up}, "injected": {"fault": r.fault, "severity_value": float(sev), "in_model": in_model}, "signals": sig, "tip": tip, "evidence": ev, "explanation": out["explanation"], "backend": out["backend"],
                         "anomaly": bool(ev["top"] != "HEALTHY" or ev["misfit"]["flag"]), "validated": out["validated"], "llm_violations": out["llm_violations"], "timing_s": {"simulate": round(t1 - t0, 2), "diagnose": round(t2 - t1, 2)}, "model": {"T": live.S["model"].T, "kscale": float(os.environ.get("KSCALE", "10")), "m": 10}})


# ---------------- phiên chạy liên tục (Start -> Apply lỗi -> báo động -> chẩn đoán -> giải thích) ----------------
class StartReq(BaseModel):
    rpm: float = Field(10500, ge=9000, le=12000)
    ae: float = 0.1905
    up: bool = False
    seed: int = 0
    backend: str = "template"
    consent: bool = False


class ApplyReq(BaseModel):
    fault: str
    severity: float = Field(0.7, ge=0.0, le=1.0)
    ramp: int = Field(6, ge=1, le=20)
    sign: int = 1


def _get(sid):
    s = stream.SESS.get(sid)
    if s is None: raise HTTPException(404, "unknown session (start a new one)")
    return s


@app.post("/api/sim/start")
def sim_start(r: StartReq):
    if min(abs(r.ae - a) for a in live.AE_VALUES) > 1e-6: raise HTTPException(400, f"ae must be one of {live.AE_VALUES}")
    if r.backend == "openrouter" and not r.consent: raise HTTPException(400, "consent is required to send the evidence bundle to OpenRouter")
    for k in [k for k, v in stream.SESS.items() if time.time() - v.created > 7200]: stream.SESS.pop(k, None)
    s = stream.Session(r.rpm, r.ae, r.up, r.seed, r.backend, r.consent); stream.SESS[s.sid] = s; return {"sid": s.sid, "n_cal": stream.N_CAL, "n_diag": stream.N_DIAG, "params": s.params}


@app.get("/api/sim/{sid}/step")
def sim_step(sid: str): return JSONResponse(_get(sid).step())


@app.post("/api/sim/{sid}/apply")
def sim_apply(sid: str, r: ApplyReq):
    s = _get(sid)
    if r.fault not in live.FAULTS or r.fault == "HEALTHY": raise HTTPException(400, "unknown fault")
    if not s.calibrated(): raise HTTPException(400, f"wait until the healthy baseline is learned ({stream.N_CAL} windows)")
    s.apply(r.fault, r.severity, r.ramp, r.sign); return {"ok": True, "tick": s.tick, "final_value": s.fault["final"], "in_model": s.fault["in_model"]}


@app.post("/api/sim/{sid}/clear")
def sim_clear(sid: str): _get(sid).clear(); return {"ok": True}


class ExplainReq(BaseModel):
    backend: str | None = None
    consent: bool | None = None


@app.post("/api/llm/test")
def llm_test():
    """Gọi thử OpenRouter bằng một câu cố định (không gửi dữ liệu máy) để biết khóa/mô hình có dùng được không."""
    import urllib.error, urllib.request
    from llm_reasoner import OpenRouterReasoner
    r = OpenRouterReasoner(allow_external=True)
    if not r.key: return {"ok": False, "model": r.model, "error": "No OPENROUTER_API_KEY found. Put OPENROUTER_API_KEY=... in prototype_kg_bn_llm/.env (no restart needed)."}
    body = {"model": r.model, "messages": [{"role": "user", "content": "Reply with the single word OK."}], "max_completion_tokens": 8}
    req = urllib.request.Request(r.URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {r.key}"}); t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp: j = json.loads(resp.read())
        return {"ok": True, "model": j.get("model", r.model), "latency_s": round(time.time() - t0, 2), "reply": (j["choices"][0]["message"]["content"] or "")[:40]}
    except urllib.error.HTTPError as e:
        msg = {401: "The key was rejected (401). Check the key.", 402: "No credit on the OpenRouter account (402).", 404: "This model name was not found or is not available for your account (404). Change OPENROUTER_MODEL.", 429: "Rate limited (429). Try again shortly."}.get(e.code, f"OpenRouter returned HTTP {e.code}.")
        return {"ok": False, "model": r.model, "error": msg}
    except Exception as e: return {"ok": False, "model": r.model, "error": f"Could not reach OpenRouter: {type(e).__name__}: {str(e)[:100]}"}


@app.post("/api/sim/{sid}/explain")
def sim_explain(sid: str, r: ExplainReq | None = None):
    s = _get(sid)
    if r is not None and r.backend in ("template", "openrouter"): s.backend = r.backend; s.consent = bool(r.consent)
    try: out = s.explain()
    except PermissionError as e: raise HTTPException(400, str(e))
    except RuntimeError as e: raise HTTPException(502, f"LLM call failed: {e}")
    if out is None: raise HTTPException(409, "no diagnosis yet")
    return out


@app.get("/api/detector")
def detector():
    p = os.path.join(HERE, "report", "detector.json"); return json.load(open(p)) if os.path.exists(p) else {}
