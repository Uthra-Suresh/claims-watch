ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=my_env

COPY requirements.txt .

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN uv venv /app/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python -r requirements.txt

COPY env/ ./env/
COPY server/ ./server/
COPY client.py .
COPY rubrics.py .
COPY inference.py .
COPY models.py .
COPY openenv.yaml .
COPY gradio_readme.md .

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/env ./env/
COPY --from=builder /app/server ./server/
COPY --from=builder /app/client.py .
COPY --from=builder /app/rubrics.py .
COPY --from=builder /app/inference.py .
COPY --from=builder /app/models.py .
COPY --from=builder /app/openenv.yaml .
COPY --from=builder /app/gradio_readme.md ./README.md

RUN mkdir -p /app/logs && chmod 777 /app/logs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 8000"]