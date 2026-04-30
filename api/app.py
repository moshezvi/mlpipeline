import logging

import pandas as pd
from flask import Flask, jsonify, request

from api.model_loader import load_model_bundle

FEATURES = ["age", "income_k", "tenure_years"]

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = app.logger

MODEL, MODEL_VERSION = load_model_bundle()


@app.before_request
def log_request():
    logger.info("Request received: %s %s", request.method, request.path)


@app.get("/health")
def health():
    logger.info("Health check successful")
    return jsonify({"status": "ok", "model_version": MODEL_VERSION}), 200


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        logger.error("Prediction request failed: invalid JSON payload")
        return jsonify({"error": "invalid JSON payload"}), 400

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        logger.error("Prediction request failed: missing fields %s", missing)
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400

    try:
        row = {feature: float(payload[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        logger.error("Prediction request failed: non-numeric feature values")
        return jsonify({"error": "feature values must be numeric"}), 400

    try:
        df = pd.DataFrame([row])
        prediction = float(MODEL.predict(df)[0])
    except Exception:
        logger.exception("Prediction request failed: model inference error")
        return jsonify({"error": "prediction failed"}), 500

    logger.info("Prediction request successful")

    return jsonify({"prediction": prediction, "model_version": MODEL_VERSION}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
