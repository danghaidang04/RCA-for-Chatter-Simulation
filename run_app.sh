#!/usr/bin/env bash
# Khởi động web app cục bộ: ./run_app.sh [cổng]   (mặc định 8000)  ->  http://127.0.0.1:8000
set -e
cd "$(dirname "$0")/prototype_kg_bn_llm"
PORT="${1:-8000}"
PY="${PYTHON:-$(command -v python3)}"
[ -f .env ] || echo "Lưu ý: chưa có prototype_kg_bn_llm/.env (copy .env.example và điền OPENROUTER_API_KEY nếu muốn dùng LLM). Không có key vẫn chạy với lời giải thích mẫu."
"$PY" -c "import fastapi, uvicorn, numba" 2>/dev/null || { echo "Thiếu thư viện: $PY -m pip install -r requirements.txt fastapi 'uvicorn[standard]'"; exit 1; }
pkill -f "uvicorn app.server" 2>/dev/null || true
( for i in $(seq 1 60); do sleep 2; curl -s "http://127.0.0.1:$PORT/api/info" >/dev/null && { echo "Sẵn sàng: http://127.0.0.1:$PORT"; command -v open >/dev/null && open "http://127.0.0.1:$PORT"; break; }; done ) &
exec "$PY" -m uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
