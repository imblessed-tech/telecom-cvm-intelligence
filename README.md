# Telecom Customer Value Management (CVM) Platform

Telecom operators process millions of transactional events daily. Each event involves a subscriber's profile, a recharge top-up, or a data session. Reviewing and optimizing subscriber retention manually is slow, expensive, and inconsistent.

This platform provides a data-driven analytics and scoring engine for these events. Given a subscriber's event history and demographics, it:
* Predicts the 30-day customer churn risk and explains the drivers using SHAP.
* Decides eligibility for custom commercial campaigns (retention, upsell, reactivation, voice top-up, loyalty).
* Segments the subscriber into K-Means behavioral segments.
* Calculates customer lifetime value (CLV) and allocates retention budgets.
* Exposes everything through a FastAPI REST API service that scores profiles and aggregates lists.
* Processes large transactional datasets in local mode using PySpark ETL.

The platform is designed to the standards expected in a production telecommunications environment — typed, tested, documented, and deployable.

---

## Table of Contents
* [Overview](#overview)
* [Architecture](#architecture)
* [Tech Stack](#tech-stack)
* [Features](#features)
* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Training the Models](#training-the-models)
* [Running the API](#running-the-api)
* [API Reference](#api-reference)
* [Running Tests](#running-tests)
* [Dashboard](#dashboard)
* [Deployment](#deployment)
* [Model Performance](#model-performance)
* [Methodology](#methodology)
* [Known Limitations](#known-limitations)
* [Author](#author)

---

## Overview
(covered above)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT / DASHBOARD                      │
│              (Browser, curl, Postman, HTML report)          │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FASTAPI REST API                          │
│         Request logging │ Timing │ CORS │ Validation        │
│               (src/cvm/api/main.py + middleware)            │
└──────┬──────────────┬───────────────┬───────────────────────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐ ┌────────────┐ ┌─────────────────┐
│  /score    │ │ /campaigns │ │    /health      │
│  /profile  │ │ /segments  │ │    /models      │
└─────┬──────┘ └────────────┘ └─────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                  DEPENDENCY LAYER                           │
│          Models loaded ONCE at startup, injected            │
│              into every route via Depends()                 │
└──────┬────────────┬────────────┬────────────┬──────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Churn   │ │ Customer │ │Campaign  │ │   CLV    │
│Predictor │ │Segmenter │ │Propensity│ │Predictor │
│ XGBoost  │ │ K-Means  │ │RF (x3)   │ │Grad Boost│
└──────────┘ └──────────┘ └──────────┘ └──────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │      ETL ENGINE PIPELINE     │
       │      (Batch Preprocessing /  │
       │       pyspark windowing)     │
       └──────────────────────────────┘
```

### Data Flow for a Single Scoring Request:
1. Client makes a `POST /api/customers/score` call.
2. FastAPI runs Pydantic validation (returns 422 if inputs violate ranges or schema).
3. Preprocessor transforms inputs (scaling features and encoding categoricals).
4. Models run in parallel:
   * `ChurnPredictor` scores probability of 30-day binary churn.
   * `CustomerSegmentation` assigns customer segment labels.
   * `PropensityModels` scores likelihood to purchase upgrades, voice top-ups, and reactivate.
   * `CLVPredictor` yields 90-day spending forecasting values.
5. The router returns a unified `ScoredResult` JSON.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.110+ | Async REST API with auto-generated docs |
| **Data** | pandas 2.0, numpy 1.26 | Data manipulation and feature engineering |
| **ML Models** | scikit-learn 1.4, XGBoost 2.0 | Ridge, Random Forest, K-Means, XGBoost Classifier, Gradient Boosting Regressor |
| **Model Persistence** | joblib | Serialise and load trained models |
| **Big Data ETL** | PySpark 3.5 | Distributed data processing and Window functions |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **Configuration** | pydantic-settings | Environment-based config management |
| **Testing** | pytest | Unit and integration testing |
| **Statistics** | scipy | KS test, drift detection |
| **Visualisation** | matplotlib | Dashboard chart generation |
| **Deployment** | Docker, Render | Containerisation and cloud hosting |
| **Language** | Python 3.11 | Core runtime |

---

## Features

### Machine Learning Pipeline
* **Churn Predictor:** XGBoost Classifier predicting binary customer churn probabilities (F1-Score, AUC-ROC).
* **Customer Segmentation:** K-Means clustering ($K=6$) grouping members by demographic and RFM features (Silhouette score).
* **Propensity Models:** RandomForest Classifiers (x3) predicting customer response probabilities for Bundle Upgrades, Voice Top-ups, and Reactivation campaigns (AUC-PR).
* **CLV Predictor:** Gradient Boosting Regressor predicting 90-day currency spend (MAE, $R^2$).

### PySpark Distributed Pipeline
* **SparkSession Engine:** Configured local SparkSession for distributed feature execution.
* **Window aggregations:** Computes rolling aggregates (recharges, data session totals).
* **LR Recharge Slope:** Computes linear regression recharge trend values natively inside Spark SQL.

### API Capabilities
* Profile details lookup, SHAP risk contribution lookup, and batch scoring endpoints.
* Campaign opportunity bases generation with reach statistics in headers.
* Service health checks and drift detection metrics.
* Auto-generated Swagger documentation at `/docs` and `/redoc`.

### Supporting Infrastructure
* Data drift detection using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test.
* Request logging and timing tracking via middleware.

---

## Project Structure

```
telecom_cvm_intelligence/
│
├── src/cvm/
│   ├── config.py                  ← all settings and paths
│   ├── main.py                    ← FastAPI app entry point (api/main.py)
│   │
│   ├── data/
│   │   ├── loader.py              ← CSV loading
│   │   ├── generator.py           ← synthetic event generator
│   │   └── preprocessor.py       ← feature cleaning and mapping
│   │
│   ├── features/
│   │   ├── rfm.py                 ← RFM calculator
│   │   └── behavioural.py         ← behavioral flags calculator
│   │
│   ├── models/
│   │   ├── base_model.py          ← abstract base ML model
│   │   ├── churn_predictor.py     ← XGBoost churn model
│   │   ├── segmentation.py        ← KMeans segmentation
│   │   ├── propensity.py          ← RandomForest classifiers
│   │   └── clv_predictor.py       ← Gradient Boosting regressor
│   │
│   ├── campaign/
│   │   └── opportunity_base.py    ← targeting suppression and channel maps
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py      ← PSI & KS calculations
│   │   └── alert_engine.py        ← alert warnings log
│   │
│   ├── visualization/
│   │   └── dashboard.py           ← visual HTML dashboards
│   │
│   └── api/
│       ├── dependencies.py        ← dependencies injection registry
│       ├── middleware.py          ← response timers and headers
│       └── routes/
│           ├── customers.py       ← profiling endpoints
│           ├── campaigns.py       ← eligibility summaries
│           ├── models.py          ← performance & drift metrics
│           └── health.py          ← service check endpoints
│
├── spark/
│   └── feature_pipeline.py        ← PySpark window ETL script
│
├── data/
│   ├── raw/                       ← inputs folder
│   └── processed/                 ← outputs feature store
│
├── models/                        ← generated joblib models
├── reports/                       ← output dashboard HTML reports
├── tests/                         ← pytest suite
│
├── train.py                       ← full training pipeline orchestrator
├── monitor.py                     ← monitoring script
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── METHODOLOGY.md
└── README.md
```

---

## Setup & Installation

### Prerequisites
* Python 3.11 or higher
* pip
* Git
* Java Runtime Environment (JRE) (required for PySpark execution)

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/telecom-cvm-intelligence.git
cd telecom-cvm-intelligence
```

### Step 2 — Create and activate a virtual environment
```bash
# Mac / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Place the Raw Data
Place the dataset `telco_churn.csv` inside `data/raw/`. The training pipeline will automatically generate `customer_events.csv` if it is missing.

### Step 5 — Configure environment
```bash
copy .env.example .env
```
The default `.env` values work for local development.

### Step 6 — Verify setup
```bash
python -c "
from src.cvm.config import settings
print('BASE_DIR:', settings.BASE_DIR)
print('TELCO_CHURN_FILE exists:', settings.TELCO_CHURN_FILE.exists())
"
```
You should see `TELCO_CHURN_FILE exists: True`.

---

## Training the Models

Run the full training pipeline with one command:
```bash
python train.py
```
This will:
* Load raw datasets and generate customer events if absent.
* Preprocess and encode demographics and transactions.
* Extract RFM scores and advanced behavioral features.
* Train Churn Predictor (XGBoost) and output SHAP summary graphs.
* Run customer segmentation ($K=6$) and save PCA plots.
* Train the propensity classifiers and predicted log-transformed CLV.
* Generate and export target opportunity lists.
* Generate the self-contained HTML dashboard report under `reports/dashboard/cvm_dashboard.html`.

---

## Running the API

### Local development
```bash
uvicorn src.cvm.api.main:app --reload --host 0.0.0.0 --port 8000
```
The `--reload` flag restarts the server automatically when you change code.

### Verify it is running
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "model_loaded_state": true,
  "version": "1.0.0",
  "timestamp": "2026-07-05T01:10:00"
}
```

### Interactive API documentation
Open in your browser:
* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

---

## API Reference

### `GET /api/customers/{customer_id}/profile`
Retrieves a customer profile and corresponding scores.

**Testing via CLI:**
* **Linux / macOS (Bash / curl):**
  ```bash
  curl -X GET "http://localhost:8000/api/customers/7590-VHVEG/profile"
  ```
* **Windows (PowerShell):**
  ```powershell
  curl -UseBasicParsing -Uri "http://localhost:8000/api/customers/7590-VHVEG/profile" -Method GET
  ```

**Response:**
```json
{
  "customer_id": "7590-VHVEG",
  "tenure_months": 1,
  "churn_risk_score": 0.7582,
  "churn_predicted": true,
  "risk_tier": "High",
  "clv_90d": 4500.5,
  "clv_tier": "Gold",
  "rfm_segment": "Champions",
  "ml_segment": "High Spend Data Users",
  "propensity_bundle": 0.8540,
  "propensity_topup": 0.6012,
  "propensity_reactivation": 0.1042,
  "lifecycle_stage": "Active"
}
```

### `GET /api/customers/{customer_id}/churn-explanation`
Retrieve top 5 risk drivers explaining churn risk scores.

**Response:**
```json
[
  {
    "feature": "monthly_charges",
    "shap_value": 0.1824,
    "direction": "increasing"
  },
  {
    "feature": "tenure_months",
    "shap_value": -0.1205,
    "direction": "decreasing"
  }
]
```

### `POST /api/customers/score`
Real-time scoring for up to 500 customers.

**Request body:**
```json
{
  "features": [
    {
      "customer_id": "cust_test_99",
      "tenure_months": 15,
      "monthly_charges": 1500.0,
      "total_charges": 22500.0,
      "recharge_count_90d": 12,
      "total_recharge_90d": 15000.0,
      "avg_recharge_amount": 1250.0,
      "days_since_last_recharge": 8,
      "data_usage_mb_90d": 1500.0,
      "data_usage_mb_30d": 500.0,
      "avg_session_mb": 40.0,
      "active_days_90d": 12,
      "app_usage_rate": 0.25,
      "recharge_trend": 0.05,
      "r_score": 4,
      "f_score": 3,
      "m_score": 4,
      "rfm_score": 11,
      "arpu": 5000.0,
      "engagement_score": 0.65,
      "declining_recharge": 0,
      "long_silence": 0,
      "low_activity": 0
    }
  ]
}
```

**Response:** Array of `ScoredResult` objects.

### `GET /api/campaigns/opportunity-base`
Retrieves prioritized target lists. Returns reach summary counts and channels inside custom headers (`X-Total-Customers`, `X-Avg-Propensity-Score`, `X-Channel-Breakdown`).

```bash
curl -i -X GET "http://localhost:8000/api/campaigns/opportunity-base?campaign_type=churn_retention&max_size=100"
```

---

## Running Tests

Run all unit tests:
```bash
pytest tests/ -v
```

---

## Dashboard

After running `train.py`, open the generated HTML report in any browser:
```bash
# Windows
start reports/dashboard/cvm_dashboard.html
```

The report contains the following visual sections:
* **Key KPI Metrics:** Total customers, high-risk counts, average ARPU.
* **Churn Analysis:** Histogram and SHAP summary plots.
* **Segmentation:** ML segments vs RFM classes.
* **Lifetime Value:** CLV tiers and R/F Heatmap.
* **Propensity & Lifecycle:** Opportunity summaries and stage funnel drop-offs.

## Deployment & Live Links

The platform is configured for production-grade deployments and is live for user testing:

* **Production Backend (Render):** The FastAPI REST API is containerized using Docker and hosted on Render. Render automatically manages build hooks and handles traffic routing to the internal Uvicorn interfaces.
* **Interactive Analytics Dashboard (GitHub Pages):** The self-contained HTML analytics dashboard is compiled dynamically by the model training pipeline and served as a static site via GitHub Pages.

### Live URLs for Testing

> [!NOTE]
> **Free Tier Spin-Down Delay:** The API is hosted on Render's free tier. If the service is inactive for 15 minutes, it automatically spins down. The first request after a sleep period will experience a **50–60 second "cold start" delay** while the container spins up and loads the machine learning models into memory. Subsequent requests will execute at normal sub-200ms latency.

* **Interactive Swagger API Documentation:** [telecom-cvm-intelligence.onrender.com/docs](https://telecom-cvm-intelligence.onrender.com/docs)
* **API Telemetry Metrics:** [telecom-cvm-intelligence.onrender.com/metrics](https://telecom-cvm-intelligence.onrender.com/metrics)
* **Static Visual Analytics Dashboard:** [imblessed-tech.github.io/telecom-cvm-intelligence/](https://imblessed-tech.github.io/telecom-cvm-intelligence/)
* **Source Code Repository:** [github.com/imblessed-tech/telecom-cvm-intelligence](https://github.com/imblessed-tech/telecom-cvm-intelligence)

---

## Model Performance

| Model | Algorithm | Metric | Value |
|---|---|---|---|
| **Churn Predictor** | XGBoost Classifier | F1 Score | `0.8554` |
| **Churn Predictor** | XGBoost Classifier | AUC-ROC | `0.9700` |
| **Segmentation** | K-Means ($K=6$) | Silhouette Score | `0.0871` |
| **CLV Predictor** | Gradient Boosting Regressor | MAE | `₦645.12` |
| **CLV Predictor** | Gradient Boosting Regressor | $R^2$ | `0.9465` |
| **Bundle Upsell** | RandomForest Classifier | AUC-PR | `1.0000` |

---

## Methodology

* **Why XGBoost for Churn?** XGBoost naturally handles high-correlation dummy columns and handles class imbalance via `scale_pos_weight` tree adjustments.
* **Why F1 thresholding?** Optimizing probability boundaries on F1 rather than accuracy prevents bias towards active customers.
* **Why PySpark windowing?** PySpark utilizes memory-efficient DAG lazy evaluation. Rolling aggregates and linear trend slopes are calculated concurrently in cluster engines.
* **Why PSI for monitoring?** Population Stability Index is the industry standard for monitoring data drift (flagging slight shifts at $\ge 0.1$ and severe drift at $\ge 0.2$).

---

## Known Limitations
* **Small Dataset:** Raw demographics represent a small sample of $7,000$ rows.
* **Synthetic events:** Ingestion events are synthetically created for demonstration.
* **Rule-based labels:** Propensity targets are engineered from rule-based thresholds.

---

## Author
* **GitHub:** [github.com/imblessed-tech](https://github.com/imblessed-tech)
