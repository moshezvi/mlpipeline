FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api

# Model-agnostic inference image: model location/version are provided at deploy time.
ENV MODEL_ARTIFACT_URI=""
ENV MODEL_VERSION=""
ENV MODEL_DIR="/app/runs/artifacts/latest"

EXPOSE 8080

CMD ["python", "api/app.py"]
