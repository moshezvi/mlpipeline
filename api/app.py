import logging
import time
import traceback
import uuid

import pandas as pd
from flask import Flask, g, jsonify, request

from api.model_loader import load_model_bundle
from api.structured_logging import log_event

FEATURES = ["age", "income_k", "tenure_years"]

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

MODEL, MODEL_VERSION = load_model_bundle()


@app.before_request
def bind_request_context():
    g.request_id = str(uuid.uuid4())
    g._req_start = time.perf_counter()


@app.after_request
def log_http_request_summary(response):
    latency_ms = round((time.perf_counter() - g._req_start) * 1000, 3)
    log_event(
        "INFO",
        event="http_request",
        request_id=g.request_id,
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )
    return response


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION}), 200


@app.post("/predict")
def predict():
    rid = g.request_id

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        log_event(
            "ERROR",
            event="predict_validation_error",
            request_id=rid,
            reason="invalid_json",
            model_version=MODEL_VERSION,
        )
        return jsonify({"error": "invalid JSON payload"}), 400

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        log_event(
            "ERROR",
            event="predict_validation_error",
            request_id=rid,
            reason="missing_fields",
            missing_fields=missing,
            model_version=MODEL_VERSION,
        )
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400

    try:
        row = {feature: float(payload[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        log_event(
            "ERROR",
            event="predict_validation_error",
            request_id=rid,
            reason="non_numeric_features",
            model_version=MODEL_VERSION,
        )
        return jsonify({"error": "feature values must be numeric"}), 400

    try:
        predict_start = time.perf_counter()
        df = pd.DataFrame([row])
        prediction = float(MODEL.predict(df)[0])
        predict_ms = round((time.perf_counter() - predict_start) * 1000, 3)
    except Exception:
        log_event(
            "ERROR",
            event="predict_inference_error",
            request_id=rid,
            model_version=MODEL_VERSION,
            traceback=traceback.format_exc(),
        )
        return jsonify({"error": "prediction failed"}), 500

    log_event(
        "INFO",
        event="predict_success",
        request_id=rid,
        model_version=MODEL_VERSION,
        predict_ms=predict_ms,
    )

    return jsonify({"prediction": prediction, "model_version": MODEL_VERSION}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
