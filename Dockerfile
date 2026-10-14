# Full live app (simulator + diagnosis + LLM explanation). For a Hugging Face Docker Space use ./deploy/make_space.sh first.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy pandas scikit-learn numba networkx matplotlib cma fastapi "uvicorn[standard]" pydantic requests
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . /app
USER user
ENV APP_WORKERS=2 PORT=7860 NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl HOME=/home/user
WORKDIR /app/prototype_kg_bn_llm
EXPOSE 7860
CMD ["sh","-c","python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
