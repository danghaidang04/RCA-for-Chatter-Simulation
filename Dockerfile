# Full live app (simulator + diagnosis + LLM explanation). Works on Hugging Face Spaces (Docker) / Render / Fly.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn numba networkx fastapi "uvicorn[standard]" pydantic requests
ENV APP_WORKERS=2 PORT=7860
WORKDIR /app/prototype_kg_bn_llm
CMD ["sh","-c","python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
