FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn websockets "pydantic>=2.0" "gradio>=4.0.0"

COPY environment.py .
COPY models.py .
COPY client.py .
COPY baseline.py .
COPY server/ ./server/
COPY README.md .

ENV ENABLE_WEB_INTERFACE=true
ENV WORKERS=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}
