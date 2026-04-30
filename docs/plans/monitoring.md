# Monitoring and Observability Design

Each section below follows **Why** (purpose), **What** (signals / metrics), and **How** (typical tools — not all required at once).

---

## API service metrics

**Why:** Prove the **inference service** is reachable, fast enough, and failing safely — this is **operations / SRE** monitoring (distinct from model accuracy).

**What:**

| Metric / signal | Role |
|-----------------|------|
| Latency **p50 / p95** for `POST /predict` | User-facing performance (SLI); tail latency catches overload or regressions. |
| **Request rate** | Traffic volume; capacity planning and anomaly detection. |
| **4xx rate** vs **5xx rate** | Client errors vs server errors (different incident classes). |
| **`/health` availability** | Liveness for load balancers and deploy verification. |
| Example **alerts** | p95 latency above threshold for a sustained window; 5xx rate above threshold. |

**How (typical AWS-oriented stack):**

- **Edge / load balancer:** **Application Load Balancer (ALB)** publishes **latency, request count, HTTP errors** to **Amazon CloudWatch** automatically.
- **Application signals:** Structured JSON logs (this repo: `api/structured_logging.py`) → **CloudWatch Logs** → **CloudWatch Logs Insights** (e.g. query `latency_ms`) or **metric filters** → **custom CloudWatch metrics** → **alarms**.
- **Optional in-process metrics:** Prometheus-style instrumentation → **AWS Distro for OpenTelemetry (ADOT)** sidecar, **Amazon Managed Prometheus**, or **statsd**-compatible agents — only if you want time series without log parsing.
- **Deployment platform:** **ECS** / **EKS** service dashboards + target health; optional **synthetic checks** (e.g. **EventBridge** + **Lambda** calling `/health`).

---

## Model performance after deployment

**Why:** Detect **model quality regression** in production after labels arrive — complements API metrics (service can be “up” but **wrong**).

**What:**

| Metric / signal | Role |
|-----------------|------|
| **Rolling RMSE** (or primary offline metric) vs **training baseline** | Grounded quality once labels exist. |
| **Prediction log** + **label join rate** | Data completeness for evaluation. |
| **Alert** | RMSE (or error rate) materially worse than baseline over an evaluation window. |

**How:**

- **Storage:** Predictions (and optional features) logged to **S3**, **warehouse** (Redshift, Snowflake, BigQuery), or **feature store**; labels ingested on a delay.
- **Compute:** **Scheduled job** (Lambda, Step Functions, Airflow, SageMaker Processing, Glue, etc.) computes RMSE on a **window** (e.g. weekly).
- **Visibility:** Emit **custom CloudWatch metrics** (`PutMetricData`), publish a **report**, or push to **Grafana** / BI; **SNS** / **Slack** for alerts.

---

## Drift checks (basic)

**Why:** Catch **input** or **output** distribution shift **before** full labels land — early warning for stale models or broken upstream data.

**What:**

| Metric / signal | Role |
|-----------------|------|
| **Input drift score** | Incoming features vs **training** reference distribution. |
| **Prediction drift score** | Distribution of **predictions** over time vs historical serving window. |
| **Threshold breach** | Triggers investigation or **retrain** playbook (not necessarily automatic retrain). |

**How:**

- **Batch (common):** Nightly job reads recent requests (from logs export or warehouse), compares to baseline → drift statistic (e.g. PSI, KS) → **metric + alert**.
- **Streaming (heavier):** **Kinesis** / **Kafka** + stream processor, or vendor **ML monitoring** — higher ops cost.

---

## What this repository implements

**Why:** Keep the take-home **runnable and reviewable** without standing up a full observability platform.

**What:**

- **Structured JSON logs** on stdout with **`request_id`**, **`latency_ms`**, **`predict_ms`**, **`status_code`**, **`model_version`**, and event types (`http_request`, `predict_success`, `predict_validation_error`, …).
- **Design-only** items above (RMSE batch, drift jobs, ALB dashboards) are **not** implemented as code here.

**How:**

- **In app:** `api/structured_logging.py`, `api/app.py`.
- **In AWS:** stdout → **ECS `awslogs`** log driver or **Fluent Bit** (EKS) → **CloudWatch Logs**; alarms/dashboards configured in **CloudWatch** (IaC or console). No CloudWatch SDK in the Flask app for baseline logging.

---

## Log shipping (AWS) — detail

**Why:** Centralize logs for **search, retention, and compliance** without coupling the app to a vendor SDK.

**What:** Same JSON events as above; **no change** to metric semantics — shipping is **transport**.

**How:** **ECS task definition** `awslogs` driver → **CloudWatch Log Group**; **EKS** **Fluent Bit** DaemonSet or sidecar → CloudWatch or OpenSearch; **IAM** and **retention** on the log group. Application remains **stdout-only**.

---

## Known limitations

**Why:** Scope and cost — a full production monitoring stack is out of bounds for this project.

**What:** No hosted APM, no automated drift/RMSE jobs, no LB metrics in CI.

**How:** Extend with the **How** rows in each section when moving to AWS accounts and staging/prod environments.

- **Label delay** still limits how quickly **true** production RMSE reflects reality.
