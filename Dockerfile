FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY env/ ./env/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
RUN mkdir -p /app/logs && chmod 777 /app/logs
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]