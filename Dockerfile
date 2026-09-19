# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app

# runtime dependencies only (no dev / eval groups); torch resolves to the CPU-only build on Linux
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# download the embedding model at build time so startup never needs the network for it
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"


FROM python:3.13-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

RUN useradd -m appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf
COPY app ./app
COPY scripts ./scripts
COPY eval/results.json ./eval/results.json
# files the results page serves: charts, architecture diagram and the technical report
COPY eval/charts ./eval/charts
COPY assets/architecture.png ./assets/architecture.png
COPY paper/experiment_report.pdf ./paper/experiment_report.pdf

# empty mount points owned by appuser, so named volumes mounted here are writable by it
RUN mkdir -p data/raw data/processed && chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
