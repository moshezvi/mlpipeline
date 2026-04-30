# Monitoring and Observability Design

## API service metrics
- Latency: p50/p95 for `POST /predict`.
- Reliability: request rate, 4xx/5xx rate, uptime of `/health`.
- Alerting:
  - p95 latency > threshold for 10 minutes
  - 5xx rate > threshold for 5 minutes

## Model performance after deployment
- Log predictions and (when available) delayed ground truth.
- Compute rolling RMSE weekly against newly labeled data.
- Alert if RMSE degrades materially compared to training baseline.

## Drift checks (basic)
- Input drift: compare incoming feature distributions vs. training baseline.
- Prediction drift: track distribution of predicted values over time.
- Trigger investigation/retraining when drift score crosses threshold.

## Minimal implementation approach
- Application logs in structured JSON with request IDs.
- Metrics emitted via middleware/counters (statsd/Prometheus style).
- Daily batch job computes drift + quality report.

## Known limitations
- No fully managed monitoring stack is implemented in this take-home.
- Label delay may limit real-time model quality tracking.
