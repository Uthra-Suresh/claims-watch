FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY env/ ./env/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY gradio_app.py .
RUN mkdir -p /app/logs && chmod 777 /app/logs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000')" || exit 1

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ENABLE_WEB_INTERFACE=true

CMD ["python", "server/app.py"]
