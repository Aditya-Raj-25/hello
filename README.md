# Explainable AI-Powered Returns Fraud Detection Dashboard

**Version:** 1.0.0 | **Language:** Python | **License:** MIT

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [Machine Learning Approach](#4-machine-learning-approach)
5. [Database Schema](#5-database-schema)
6. [UML Component Diagram](#6-uml-component-diagram)
7. [ML Pipeline Diagram](#7-ml-pipeline-diagram)
8. [Core Dashboard Metrics](#8-core-dashboard-metrics)
9. [User Risk Categorization](#9-user-risk-categorization)
10. [Logs and Monitoring](#10-logs-and-monitoring)
11. [Why This Project Stands Out](#11-why-this-project-stands-out)
12. [Technology Stack](#12-technology-stack)
13. [Project Structure](#13-project-structure)
14. [Setup and Installation](#14-setup-and-installation)
15. [API Reference](#15-api-reference)
16. [Future Enhancements](#16-future-enhancements)

---

## 1. Project Overview

### The Problem

E-commerce platforms lose billions of dollars annually to return fraud — a class of abuse that is financially damaging, operationally costly to investigate, and extremely difficult to detect using traditional rule-based systems.

Common fraud patterns include:

| Fraud Type                   | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| Serial Returning             | Users who return nearly every purchase, often at high monetary value        |
| Wardrobing                   | Items bought for temporary use (events, travel) then returned               |
| High-Value Item Abuse        | Deliberate targeting of expensive SKUs for fraudulent returns               |
| Geolocation Mismatch         | Returns purchased in one region but submitted from a different location     |
| Return Velocity Abuse        | Concentrated burst of returns in a short time window to exploit policies    |
| Refund Pattern Manipulation  | Rotating refund methods or accounts to bypass detection                     |

### Why Rule-Based Systems Fail

Hard-coded rules (e.g., "flag if return rate > 30%") produce high false-positive rates, are trivially evaded by fraudsters who stay just below thresholds, and cannot adapt to novel behavioral patterns. They also provide no explanation for why a user was flagged, creating legal and operational risk.

### Why This Approach Works

This system uses unsupervised anomaly detection (Isolation Forest) to model legitimate user behavior from raw transaction logs and surface statistical outliers — without requiring labeled fraud examples. SHAP (SHapley Additive exPlanations) decomposes each prediction into per-feature contributions, enabling investigators to validate flags rather than blindly act on them.

### Business Impact

- Reduces fraudulent refund payouts by identifying high-risk users before refunds are processed
- Minimizes false positives by routing medium-risk users to human review rather than automated blocking
- Provides audit-grade explainability to satisfy compliance and dispute resolution requirements
- Quantifies total fraud exposure and recovery as actionable financial metrics

---

## 2. Key Features

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| CSV Transaction Log Ingestion    | Upload raw order/return logs; pipeline triggers automatically               |
| Real-Time User Search            | Lookup any user by ID or email to retrieve their full risk profile          |
| Risk Score (0-100)               | Normalized fraud probability score derived from Isolation Forest output     |
| Risk Band Classification         | Automatic Low / Medium / High tiering for triage workflows                  |
| Total Financial Loss Calculation | Aggregate refund exposure across all high-risk flagged users                |
| Total Loss Recovery Tracking     | Tracks refunds blocked post-flagging as recovered value                     |
| Fraud vs. Legitimate Distribution| Visual breakdown of risk band proportions across the user base              |
| Top Fraud Risk Factors           | SHAP-derived feature contributions ranked by impact per user                |
| User Investigation Panel         | Side-by-side view of risk profile, behavioral history, and audit trail      |
| Behavioral Logs and Timeline     | Chronological view of return events, amounts, and triggered flags           |
| Risk Sensitivity Threshold Slider| Admin control to tune contamination rate without retraining the model       |
| Return Frequency Tracking        | Computed ratio of total returns to total purchases per user                 |
| Account Age Analysis             | Flags newly created accounts with high return activity                      |
| High-Value Item Detection        | Tracks returns of items exceeding the configurable high-value threshold     |

---

## 3. System Architecture

The system is built entirely in Python and organized into four horizontal layers. Each layer communicates through well-defined interfaces.

```
+-------------------------------------------------------------+
|                        FRONTEND LAYER                       |
|  Streamlit / Plotly Dash                                    |
|  - Risk Score Overview Panel                                |
|  - Fraud Distribution Charts (Low / Medium / High)          |
|  - User Search and Investigation Panel                      |
|  - Behavioral Timeline View                                 |
|  - SHAP Feature Contribution Breakdown                      |
|  - Risk Threshold Slider (admin control)                    |
+------------------------------+------------------------------+
                               |
                          HTTP / REST
                               |
+------------------------------v------------------------------+
|                        BACKEND LAYER                        |
|  FastAPI / Flask Application Server                         |
|  - /api/upload      → CSV ingestion and validation         |
|  - /api/users       → User risk profile retrieval          |
|  - /api/scores      → Aggregated score endpoint            |
|  - /api/explain     → SHAP explanation retrieval           |
|  - /api/logs        → Audit log access                     |
|  - /api/threshold   → Sensitivity configuration            |
|  Auth Module        → JWT-based token verification         |
|  Risk Score Service → Triggers ML pipeline on upload       |
|  Logging Service    → Stores all investigator actions      |
+------------------------------+------------------------------+
                               |
                      Internal function call
                               |
+------------------------------v------------------------------+
|                   MACHINE LEARNING LAYER                    |
|  Feature Engineering Module (Pandas)                        |
|  Isolation Forest Model (scikit-learn)                      |
|  Logistic Regression Layer (optional supervised)            |
|  Risk Score Normalizer  → Maps scores to 0-100             |
|  Risk Band Classifier   → Low / Medium / High              |
|  Explainability Module  → SHAP TreeExplainer               |
+------------------------------+------------------------------+
                               |
                     Read / Write operations
                               |
+------------------------------v------------------------------+
|                         DATA LAYER                          |
|  Raw CSV Store       → Uploaded transaction files           |
|  Feature Store       → Engineered feature vectors          |
|  PostgreSQL / MongoDB → Users, Transactions, FraudScores   |
|  Logs Table          → Full audit trail and system events  |
+-------------------------------------------------------------+
```

### Data Flow

```
User Action
  → Python Frontend (Streamlit / Dash)
    → Backend API (FastAPI / Flask) — auth, routing, validation
      → ML Engine — feature engineering, scoring, SHAP
        → Database — persist scores, logs, feature vectors
          → Backend — format structured response
            → Dashboard — render updated metrics and panels
```

---

## 4. Machine Learning Approach

### 4.1 Feature Engineering

All features are derived from raw transaction logs per user. No external data sources are required.

| Feature                      | Formula / Logic                                             |
|------------------------------|-------------------------------------------------------------|
| Return Frequency             | total_returns / total_purchases                             |
| Return Velocity (30d)        | COUNT(returns) WHERE return_date >= NOW() - 30 days         |
| Average Time-to-Return       | AVG(return_date - purchase_date) in days                    |
| High-Value Item Ratio        | returns_above_threshold / total_returns                     |
| Category Repetition Score    | Entropy of product categories among returned items          |
| Account Age                  | Days elapsed since account creation date                    |
| Geolocation Mismatch Count   | COUNT(returns where return_region != purchase_region)       |
| Refund Method Consistency    | Variance in refund_destination across return transactions   |

### 4.2 Anomaly Detection — Isolation Forest

Isolation Forest is used as the primary detection model. It assigns each user an anomaly score based on how easily they can be isolated from the rest of the distribution. Fraudulent users, who exhibit behavioral extremes, are isolated in fewer splits.

Key configuration parameters:

```python
IsolationForest(
    n_estimators=200,
    contamination=0.05,   # Tunable via threshold slider
    max_features=1.0,
    random_state=42
)
```

The `contamination` parameter directly corresponds to the expected fraud prevalence and is the primary lever exposed to administrators via the sensitivity threshold slider.

### 4.3 Optional Supervised Layer — Logistic Regression

If historical fraud labels exist (from past investigations), a Logistic Regression classifier can be trained on top of engineered features to provide a secondary probability estimate. Its output is blended with the Isolation Forest score.

**Handling Class Imbalance:**
- SMOTE (Synthetic Minority Oversampling Technique) for minority class augmentation
- `class_weight='balanced'` argument in scikit-learn estimators

### 4.4 Risk Score Formula

The final risk score (0–100) is computed as a weighted combination of normalized sub-scores:

```
Risk Score = (
    0.35 * normalized_anomaly_score
  + 0.25 * return_frequency_score
  + 0.20 * high_value_ratio_score
  + 0.10 * geolocation_risk_score
  + 0.10 * timing_anomaly_score
)
* 100
```

All sub-scores are individually min-max normalized to [0, 1] before combination.

### 4.5 Explainability — SHAP

SHAP (SHapley Additive exPlanations) is applied to break down each user's risk score into per-feature contributions:

```python
import shap

explainer = shap.TreeExplainer(isolation_forest_model)
shap_values = explainer.shap_values(user_feature_vector)
top_factors = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
```

The top contributing factors are surfaced in the User Investigation Panel with human-readable labels (e.g., "High return velocity in past 30 days", "Geolocation mismatch detected").

---

## 5. Database Schema

### Entity Relationship Diagram

```
+----------+          +------------------+          +---------+
|  Users   | 1      * |  Transactions    | 1      0-1|  Returns|
|----------|--------->|------------------|---------->|---------|
| user_id  |          | txn_id           |           | ret_id  |
| email    |          | user_id (FK)     |           | txn_id  |
| name     |          | item_id          |           | reason  |
| region   |          | item_value       |           | amount  |
| acct_age |          | category         |           | ret_dt  |
| created_at          | purchase_date    |           | method  |
+----+-----+          | status           |           +---------+
     |                +------------------+
     | 1
     |
     +-------> 1  +------------------+
     |             |  FraudScores     |
     |             |------------------|
     |             | score_id         |
     |             | user_id (FK)     |
     |             | risk_score       |
     |             | risk_band        |
     |             | feature_vector   |
     |             | shap_values      |
     |             | computed_at      |
     |             +------------------+
     |
     +-------> *  +------------------+
                  |  Logs            |
                  |------------------|
                  | log_id           |
                  | user_id (FK)     |
                  | action           |
                  | actor            |
                  | timestamp        |
                  | detail           |
                  +------------------+
```

### Relationship Summary

| Relationship             | Cardinality          |
|--------------------------|----------------------|
| User to Transactions     | One-to-Many          |
| Transaction to Return    | One-to-Optional-One  |
| User to FraudScore       | One-to-One           |
| User to Logs             | One-to-Many          |

---

## 6. UML Component Diagram

```
+-------------------------+
|       Frontend          |
|  (Streamlit / Dash)     |
|  - Dashboard UI         |
|  - Search Panel         |
|  - Threshold Slider     |
+----------+--------------+
           |
           | HTTP REST API calls
           |
+----------v--------------+
|      Backend API        |
|  (FastAPI / Flask)      |
|  - /upload              |
|  - /users               |
|  - /scores              |
|  - /explain             |
|  - /logs                |
|  - /threshold           |
+----+----------+---------+
     |          |
     |          | Triggers pipeline
     |          |
     |   +------v-----------+
     |   |    ML Engine     |
     |   |------------------|
     |   | Feature Engineer |
     |   | Isolation Forest |
     |   | Risk Normalizer  |
     |   | SHAP Explainer   |
     |   +------+-----------+
     |          |
     | Reads /  | Writes
     | Writes   |
     +----------v-----------+
     |       Database       |
     |  (PostgreSQL / CSV)  |
     |----------------------|
     | Users                |
     | Transactions         |
     | Returns              |
     | FraudScores          |
     | Logs                 |
     +----------------------+
```

---

## 7. ML Pipeline Diagram

```
+---------------------------+
|   Raw CSV Transaction Log |
+-------------+-------------+
              |
              v
+---------------------------+
|   Data Validation         |
|   - Schema check          |
|   - Null removal          |
|   - Timestamp normalization|
|   - Deduplication         |
+-------------+-------------+
              |
              v
+---------------------------+
|   Feature Engineering     |
|   - Return Frequency      |
|   - Return Velocity       |
|   - Avg Time-to-Return    |
|   - High-Value Ratio      |
|   - Account Age           |
|   - Geolocation Mismatch  |
|   - Refund Consistency    |
+-------------+-------------+
              |
              v
+---------------------------+
|   Model Training          |
|   Isolation Forest        |
|   (contamination tunable) |
+-------------+-------------+
              |
              v
+---------------------------+
|   Anomaly Score Output    |
|   (decision_function())   |
+-------------+-------------+
              |
              v
+---------------------------+
|   Risk Score (0-100)      |
|   Weighted normalization  |
|   of sub-scores           |
+-------------+-------------+
              |
              v
+---------------------------+
|   Explainability (SHAP)   |
|   Per-feature contribution|
|   Top 5 fraud factors     |
+-------------+-------------+
              |
              v
+---------------------------+
|   Dashboard Visualization |
|   - Risk panels           |
|   - User profiles         |
|   - Financial metrics     |
+---------------------------+
```

---

## 8. Core Dashboard Metrics

| Metric                   | Computation Logic                                                          |
|--------------------------|----------------------------------------------------------------------------|
| Total Transactions Scanned | COUNT(*) across the Transactions table after CSV ingestion              |
| Total Fraud Detected     | COUNT(users) WHERE risk_band = 'High'                                      |
| Total Financial Loss     | SUM(return_amount) WHERE user.risk_band = 'High'                           |
| Total Loss Recovered     | SUM(return_amount) WHERE refund_blocked = TRUE AND user.risk_band = 'High' |
| Fraud Rate (%)           | (COUNT(High Risk Users) / COUNT(All Users)) * 100                          |
| Risk Accuracy            | Model Precision = TP / (TP + FP) evaluated on labeled validation set        |

> Without ground-truth labels, Financial Loss is estimated as the full refund exposure of high-risk users. Loss Recovered tracks refunds explicitly reversed or blocked post-investigation.

---

## 9. User Risk Categorization

### Risk Bands

| Band        | Score Range | Recommended Action                                          |
|-------------|-------------|-------------------------------------------------------------|
| Low Risk    | 0 - 40      | No immediate action; routine monitoring                     |
| Medium Risk | 41 - 70     | Flag for investigator review before processing refund       |
| High Risk   | 71 - 100    | Automatic hold on refund; escalate for manual investigation |

### Investigator Workflow

1. The investigator accesses the dashboard and filters users by risk band.
2. For each medium or high-risk user, the Investigation Panel displays:
   - The risk score and band
   - The top 5 SHAP-derived fraud factors with magnitude
   - The full behavioral timeline (return dates, amounts, product categories)
   - Geolocation data overlaid on purchase vs. return origin
3. The investigator approves or denies the refund and logs the action, which is written to the Logs table.
4. The admin can adjust the risk sensitivity threshold slider to recalibrate flagging aggressiveness without retraining the model. A lower contamination value reduces sensitivity; a higher value increases it.

---

## 10. Logs and Monitoring

All system and investigator actions are persisted to the Logs table with a timestamp, actor ID, and action detail.

| Log Event                        | Trigger                                                      |
|----------------------------------|--------------------------------------------------------------|
| CSV Upload                       | Admin uploads a new transaction log file                     |
| Pipeline Run Complete            | Feature engineering and scoring pipeline finishes            |
| User Profile Viewed              | Investigator opens a user's investigation panel              |
| Refund Approved / Denied         | Investigator takes action on a flagged user                  |
| Threshold Adjusted               | Admin changes the risk sensitivity value                     |
| Score Recomputed                 | Risk scores refreshed after threshold change or new data     |

The behavioral timeline per user provides a chronological view of all return events, amounts, and the timestamps at which any risk flags were triggered. This timeline is the primary audit artifact in dispute resolution scenarios.

---

## 11. Why This Project Stands Out

**Hybrid Detection Architecture**
The system combines unsupervised anomaly detection (Isolation Forest) for novel fraud patterns with an optional supervised classifier (Logistic Regression) when historical labels are available. Neither approach alone handles both labeled and unlabeled scenarios.

**Explainable AI by Design**
SHAP integration is not cosmetic. Every risk score is decomposed into feature-level contributions with human-readable labels. Investigators never act on a black-box output — they see exactly why a user was flagged.

**Sensitivity Tuning Without Retraining**
The contamination parameter is exposed as an admin-controlled slider. Changing the threshold recalibrates the scoring cutoffs in real time without requiring model retraining, enabling rapid operational adjustment during high-fraud periods.

**Enterprise-Style Dashboard**
The dashboard is built for operations teams, not data scientists. Risk overview, user search, investigation panel, financial metrics, and audit logs are all surfaced in a single interface designed for practical triage workflows.

**Fairness-Aware Scoring**
The risk score formula explicitly avoids using demographic features. All inputs are behavioral and transactional, reducing the risk of biased flagging across user segments.

---

## 12. Technology Stack

| Layer        | Technology                                                    |
|--------------|---------------------------------------------------------------|
| Frontend     | Streamlit or Plotly Dash (Python)                             |
| Backend      | FastAPI or Flask (Python), JWT Authentication                 |
| ML Engine    | scikit-learn (Isolation Forest, Logistic Regression), SHAP    |
| Data Processing | Pandas, NumPy                                              |
| Imbalanced Data | imbalanced-learn (SMOTE)                                   |
| Database     | PostgreSQL (production) or CSV flat files (demo mode)         |
| Deployment   | Docker Compose (backend + frontend + database)                |

---

## 13. Project Structure

```
fraud-detection-dashboard/
|
|-- frontend/
|   |-- app.py                   # Streamlit / Dash entry point
|   |-- components/
|   |   |-- risk_overview.py     # KPI summary panel
|   |   |-- user_search.py       # User lookup interface
|   |   |-- investigation.py     # Risk profile + SHAP breakdown
|   |   |-- timeline.py          # Behavioral event timeline
|   |   |-- threshold_slider.py  # Admin sensitivity control
|
|-- backend/
|   |-- main.py                  # FastAPI / Flask application entry
|   |-- routers/
|   |   |-- upload.py            # /api/upload
|   |   |-- users.py             # /api/users
|   |   |-- scores.py            # /api/scores
|   |   |-- explain.py           # /api/explain
|   |   |-- logs.py              # /api/logs
|   |   |-- threshold.py         # /api/threshold
|   |-- services/
|   |   |-- risk_service.py      # Pipeline orchestration
|   |   |-- auth.py              # JWT handling
|   |   |-- logger.py            # Audit log writes
|   |-- ml/
|   |   |-- feature_engineering.py  # Pandas feature computation
|   |   |-- model.py                # Isolation Forest wrapper
|   |   |-- scorer.py               # Normalization + band assignment
|   |   |-- explainer.py            # SHAP integration
|   |-- models/
|   |   |-- schemas.py           # Pydantic request/response models
|   |   |-- db.py                # Database connection
|   |-- requirements.txt
|
|-- data/
|   |-- raw/                     # Uploaded CSV files
|   |-- processed/               # Engineered feature vectors
|   |-- sample_transactions.csv  # Demo dataset
|
|-- docker-compose.yml
|-- .env.example
|-- README.md
```

---

## 14. Setup and Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (or use CSV demo mode)
- Docker and Docker Compose (optional but recommended)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/your-org/fraud-detection-dashboard.git
cd fraud-detection-dashboard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your database credentials and JWT secret

# Run the backend
uvicorn backend.main:app --reload --port 8000

# Run the frontend (separate terminal)
streamlit run frontend/app.py
```

### Docker Setup

```bash
docker-compose up --build
```

The dashboard will be accessible at `http://localhost:8501`.
The API will be accessible at `http://localhost:8000`.
API documentation (auto-generated) at `http://localhost:8000/docs`.

### CSV Demo Mode

To run without a database, set `DEMO_MODE=true` in `.env`. The system will read from and write to CSV files in the `data/` directory.

---

## 15. API Reference

| Method | Endpoint           | Description                                         |
|--------|--------------------|-----------------------------------------------------|
| POST   | /api/upload        | Upload CSV transaction log; triggers ML pipeline    |
| GET    | /api/users         | List all users with risk scores and bands           |
| GET    | /api/users/{id}    | Retrieve risk profile for a specific user           |
| GET    | /api/scores        | Aggregated risk score summary statistics            |
| GET    | /api/explain/{id}  | SHAP feature contributions for a specific user      |
| GET    | /api/logs          | Retrieve audit logs (admin only)                    |
| POST   | /api/threshold     | Update risk sensitivity threshold value             |

All endpoints require a valid JWT Bearer token in the `Authorization` header, except `/api/upload` in demo mode.

---

## 16. Future Enhancements

| Enhancement                         | Description                                                          |
|-------------------------------------|----------------------------------------------------------------------|
| Real-Time Streaming Detection       | Kafka + Faust integration for per-transaction scoring at ingest time |
| Graph-Based Fraud Network Detection | NetworkX or Neo4j to detect coordinated fraud rings across accounts  |
| Sequence Modeling                   | LSTM or Transformer on return event sequences for temporal patterns  |
| Auto Threshold Optimization         | Bayesian optimization of contamination parameter using feedback data |
| Self-Supervised Pretraining         | Contrastive learning on transaction embeddings for richer features   |
| Multi-Tenant Support                | Isolated pipelines and dashboards per merchant or business unit      |

---

*Built for hackathon submission. System designed for 1-2 week implementation sprint.*
*All ML components are production-ready and replaceable with minimal interface changes.*
