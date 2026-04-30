FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY runs/artifacts ./runs/artifacts

# Local verification image: bundles latest local artifacts for quick /predict checks.
ENV MODEL_DIR=/app/runs/artifacts/latest
EXPOSE 8080

CMD ["python", "api/app.py"]
