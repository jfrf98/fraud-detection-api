# Fraud Detection API

Machine Learning API for detecting fraudulent credit card transactions using a trained Random Forest model, FastAPI, Docker, and basic monitoring endpoints.

This project was built as part of a Machine Learning Engineering portfolio. The goal is not only to train a model, but to create a deployable ML service with inference, validation, logging, monitoring, and containerization.

---

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small percentage of total transactions.

In this project, I built an end-to-end ML system that:

* Loads and analyzes a real-world fraud dataset.
* Trains and compares machine learning models.
* Selects a production model based on business-oriented metrics.
* Serves predictions through a FastAPI REST API.
* Adds risk-based decision rules.
* Logs predictions for auditability.
* Exposes monitoring endpoints.
* Runs inside a Docker container.

---

## Business Problem

In fraud detection, the cost of missing a fraudulent transaction can be high. Therefore, accuracy is not the most relevant metric.

The main business goal is to:

> Detect as many fraudulent transactions as possible while keeping false positives under control.

This project focuses on metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

Special attention is given to **recall**, because false negatives represent fraud cases that were not detected.

---

## Dataset

Dataset used:

**Credit Card Fraud Detection Dataset**

The dataset contains anonymized credit card transactions, where:

* `Class = 0` means normal transaction.
* `Class = 1` means fraudulent transaction.

Dataset characteristics:

* Rows: 284,807
* Features: 30
* Target: `Class`
* Fraud percentage: approximately 0.17%

Because of the strong class imbalance, accuracy alone is not a reliable metric.

---

## Model Development

Two models were evaluated:

### Logistic Regression Baseline

A Logistic Regression model was trained as the baseline.

Because Logistic Regression is sensitive to feature scaling, `StandardScaler` was applied.

Results for class `1`:

| Metric    |  Score |
| --------- | -----: |
| Precision |   0.87 |
| Recall    |   0.62 |
| F1-score  |   0.73 |
| ROC-AUC   | 0.9782 |

---

### Random Forest Classifier

A Random Forest model was trained as the improved model.

Tree-based models do not require feature scaling, so the model uses the original input features.

Results for class `1`:

| Metric    |  Score |
| --------- | -----: |
| Precision |   0.97 |
| Recall    |   0.80 |
| F1-score  |   0.88 |
| ROC-AUC   | 0.9476 |

Confusion matrix:

|               | Predicted Normal | Predicted Fraud |
| ------------- | ---------------: | --------------: |
| Actual Normal |            56862 |               2 |
| Actual Fraud  |               20 |              78 |

---

## Model Selection

Although Logistic Regression achieved a higher ROC-AUC, Random Forest was selected as the production model because it achieved better recall and F1-score.

This means Random Forest detected more fraudulent transactions, which is better aligned with the business goal of reducing fraud losses.

Selected production model:

```text
RandomForestClassifier
```

Model version:

```text
v2-random-forest
```

---

## Risk Decision Logic

The API does not only return a binary prediction. It also converts the fraud probability into a business decision.

| Fraud Probability | Risk Level | Recommended Action |
| ----------------: | ---------- | ------------------ |
|          `< 0.30` | Low        | Approve            |
|     `0.30 - 0.69` | Medium     | Manual Review      |
|         `>= 0.70` | High       | Block              |

Example response:

```json
{
  "prediction": 1,
  "label": "fraud",
  "fraud_probability": 0.82,
  "risk_level": "high",
  "recommended_action": "block",
  "model_version": "v2-random-forest",
  "threshold_used": 0.5
}
```

---

## API Endpoints

### `GET /`

Basic root endpoint.

Response:

```json
{
  "message": "Fraud Detection API is running"
}
```

---

### `GET /health`

Checks whether the API and model are running correctly.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "model_version": "v2-random-forest"
}
```

---

### `GET /model-info`

Returns model metadata.

Example response:

```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "v2-random-forest",
  "features_used": 30,
  "threshold": 0.5
}
```

---

### `POST /predict`

Receives transaction features and returns a fraud prediction.

Input format:

```json
{
  "features": [
    406,
    -2.312227,
    1.951992,
    -1.609851,
    3.997906,
    -0.522188,
    -1.426545,
    -2.537387,
    1.391657,
    -2.770089,
    -2.772272,
    3.202033,
    -2.899907,
    -0.595222,
    -4.289254,
    0.389724,
    -1.140747,
    -2.830056,
    -0.016822,
    0.416956,
    0.126911,
    0.517232,
    -0.035049,
    -0.465211,
    0.320198,
    0.044519,
    0.17784,
    0.261145,
    -0.143276,
    0
  ]
}
```

Example response:

```json
{
  "prediction": 1,
  "label": "fraud",
  "fraud_probability": 0.82,
  "risk_level": "high",
  "recommended_action": "block",
  "model_version": "v2-random-forest",
  "threshold_used": 0.5
}
```

---

### `GET /stats`

Returns operational counts based on prediction logs.

Example response:

```json
{
  "total_predictions": 4,
  "fraud_predictions": 2,
  "normal_predictions": 2,
  "high_risk": 2,
  "medium_risk": 0,
  "low_risk": 2
}
```

---

### `GET /metrics`

Returns operational percentages based on prediction logs.

Example response:

```json
{
  "fraud_rate": 50.0,
  "high_risk_rate": 50.0,
  "approval_rate": 50.0,
  "manual_review_rate": 0.0,
  "block_rate": 50.0
}
```

---

### `GET /logs/recent`

Returns recent prediction logs.

Optional query parameter:

```text
limit
```

Example:

```text
/logs/recent?limit=5
```

---

## Project Structure

```text
fraud-detection-api/
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── utils.py
│   │   └── config.py
│   │
│   └── models/
│       ├── fraud_model.pkl
│       └── scaler.pkl
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Docker
* Git / GitHub

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/jfrf98/fraud-detection-api.git
cd fraud-detection-api
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Windows CMD:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## How to Run with Docker

### 1. Build the image

```bash
docker build -t fraud-detection-api .
```

### 2. Run the container

```bash
docker run -p 8000:8000 fraud-detection-api
```

Open:

```text
http://localhost:8000/docs
```

---

## Input Validation

The API validates:

* The request must contain exactly 30 features.
* All features must be numeric.
* Values must be finite.
* Invalid inputs return controlled error messages.

Example error:

```json
{
  "detail": "The transaction must contain exactly 30 features."
}
```

---

## Logging and Monitoring

Each prediction is logged into:

```text
logs/predictions.csv
```

Logged fields:

* timestamp
* prediction
* fraud_probability
* risk_level
* recommended_action
* model_version

The API also exposes monitoring endpoints:

* `/stats`
* `/metrics`
* `/logs/recent`

This adds a basic observability layer for inference monitoring and debugging.

---

## Key ML Engineering approaches

* Handling imbalanced classification.
* Choosing metrics based on business goals.
* Comparing baseline and improved models.
* Model serialization with Joblib.
* Model serving with FastAPI.
* Input validation with Pydantic.
* Risk-based decision rules.
* Inference logging.
* Operational monitoring.
* Dockerized deployment-ready API.
* Clean project structure.
* Git/GitHub version control.

---

## Next Steps

Planned improvements:

* Deploy API publicly using Render.
* Add automated tests.
* Add CI/CD with GitHub Actions.
* Add a Streamlit dashboard for monitoring.
* Add model explainability with SHAP.
* Add batch prediction endpoint.

---

## Author

Developed by Francisco Fabre as part of a Machine Learning Engineering portfolio project.
