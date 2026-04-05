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
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "gradio_app.py"]
