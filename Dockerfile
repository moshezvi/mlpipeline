FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY artifacts ./artifacts

ENV MODEL_DIR=/app/artifacts/latest
EXPOSE 8080

CMD ["python", "api/app.py"]
