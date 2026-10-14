"""Entry point for a Hugging Face Space with SDK 'gradio' (free): Spaces simply run this file, which starts the FastAPI app on port 7860.
It first joins the data file that was shipped in parts (Hugging Face rejects plain git files above 10 MiB)."""
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(HERE, "cutting_sim", "output"); REAL = os.path.join(OUTD, "real_windows.npz")
if not os.path.exists(REAL):
    import base64
    parts = sorted(glob.glob(os.path.join(OUTD, "real_windows.b64.part_*")))
    raw = base64.b64decode("".join(open(p).read().replace("\n", "") for p in parts))
    want = open(os.path.join(OUTD, "real_windows.sha256")).read().split()[0]
    if hashlib.sha256(raw).hexdigest() != want: raise SystemExit("data file checksum mismatch after joining the parts")
    open(REAL, "wb").write(raw)
    for p in parts: os.remove(p)
for k, v in {"APP_WORKERS": "2", "MAX_SESSIONS": "8", "LLM_PER_IP_HOUR": "0", "LLM_PER_DAY": "0", "NUMBA_CACHE_DIR": "/tmp/numba", "MPLCONFIGDIR": "/tmp/mpl"}.items(): os.environ.setdefault(k, v)
sys.path.insert(0, os.path.join(HERE, "prototype_kg_bn_llm")); os.chdir(os.path.join(HERE, "prototype_kg_bn_llm"))
import uvicorn  # noqa: E402
from app.server import app  # noqa: E402,F401

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
