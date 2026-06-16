import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_risk_decision(probability: float) -> tuple[str, str]:
    if probability < 0.30:
        return "low", "approve"
    elif probability < 0.70:
        return "medium", "manual_review"
    else:
        return "high", "block"


def log_prediction(
    probability: float,
    prediction: int,
    risk_level: str,
    action: str,
    model_version: str,
    log_file: Path
) -> None:
    file_exists = log_file.exists()

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "fraud_probability",
                "risk_level",
                "recommended_action",
                "model_version"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            int(prediction),
            float(probability),
            risk_level,
            action,
            model_version
        ])


def get_prediction_stats(log_file: Path) -> dict:
    if not log_file.exists():
        return {
            "total_predictions": 0,
            "fraud_predictions": 0,
            "normal_predictions": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0
        }

    df = pd.read_csv(log_file)

    return {
        "total_predictions": len(df),
        "fraud_predictions": int((df["prediction"] == 1).sum()),
        "normal_predictions": int((df["prediction"] == 0).sum()),
        "high_risk": int((df["risk_level"] == "high").sum()),
        "medium_risk": int((df["risk_level"] == "medium").sum()),
        "low_risk": int((df["risk_level"] == "low").sum())
    }

def get_prediction_metrics(log_file: Path) -> dict:
    if not log_file.exists():
        return {
            "fraud_rate": 0.0,
            "high_risk_rate": 0.0,
            "approval_rate": 0.0,
            "manual_review_rate": 0.0,
            "block_rate": 0.0
        }

    df = pd.read_csv(log_file)

    total = len(df)

    if total == 0:
        return {
            "fraud_rate": 0.0,
            "high_risk_rate": 0.0,
            "approval_rate": 0.0,
            "manual_review_rate": 0.0,
            "block_rate": 0.0
        }

    return {
        "fraud_rate": round(
            (df["prediction"] == 1).mean() * 100, 2
        ),
        "high_risk_rate": round(
            (df["risk_level"] == "high").mean() * 100, 2
        ),
        "approval_rate": round(
            (df["recommended_action"] == "approve").mean() * 100, 2
        ),
        "manual_review_rate": round(
            (df["recommended_action"] == "manual_review").mean() * 100, 2
        ),
        "block_rate": round(
            (df["recommended_action"] == "block").mean() * 100, 2
        )
    }

def get_recent_predictions(
    log_file: Path,
    limit: int = 10
) -> dict:

    if not log_file.exists():
        return {"recent_predictions": []}

    df = pd.read_csv(log_file)

    recent = (
        df.tail(limit)
        .iloc[::-1]
        .to_dict(orient="records")
    )

    return {
    "total_returned": len(recent),
    "recent_predictions": recent
}