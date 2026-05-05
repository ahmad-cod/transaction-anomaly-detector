"""
FastAPI — Transaction Risk Scoring Endpoint
============================================
Run:
    uvicorn api:app --reload

Docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import time

from anomaly_detector import (
    TransactionAnomalyDetector,
    generate_synthetic_transactions,
)


# ─────────────────────────────────────────────
# 1. Model lifecycle — train once at startup
# ─────────────────────────────────────────────

detector: TransactionAnomalyDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train the detector on synthetic data at startup, release at shutdown."""
    global detector
    print("Training anomaly detector on synthetic data...")
    df = generate_synthetic_transactions(n_normal=2000, n_fraud=100)
    detector = TransactionAnomalyDetector()
    detector.fit(df)
    print("Detector ready.")
    yield
    detector = None


# ─────────────────────────────────────────────
# 2. App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Transaction Anomaly Detector",
    description="Accepts a transaction and returns a risk score + fraud decision.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# 3. Request / Response schemas
# ─────────────────────────────────────────────

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    txn_freq_1h: int = Field(..., ge=0, description="Number of txns in past hour")
    txn_freq_24h: int = Field(..., ge=0, description="Number of txns in past 24h")
    country_match: int = Field(..., ge=0, le=1, description="1 if home country, 0 if foreign")
    merchant_risk: int = Field(..., ge=0, le=2, description="Merchant risk tier: 0=low, 1=med, 2=high")
    declined_prev: int = Field(..., ge=0, description="Count of previously declined txns")
    days_since_first: int = Field(default=365, ge=0, description="Days since account first txn")

    @model_validator(mode="after")
    def freq_sanity(self):
        if self.txn_freq_1h > self.txn_freq_24h:
            raise ValueError("txn_freq_1h cannot exceed txn_freq_24h")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Normal transaction",
                    "value": {
                        "amount": 45.99,
                        "hour": 14,
                        "txn_freq_1h": 1,
                        "txn_freq_24h": 3,
                        "country_match": 1,
                        "merchant_risk": 0,
                        "declined_prev": 0,
                        "days_since_first": 400,
                    },
                },
                {
                    "summary": "Suspicious transaction",
                    "value": {
                        "amount": 7500.00,
                        "hour": 3,
                        "txn_freq_1h": 15,
                        "txn_freq_24h": 40,
                        "country_match": 0,
                        "merchant_risk": 2,
                        "declined_prev": 3,
                        "days_since_first": 1,
                    },
                },
            ]
        }
    }


class RiskResponse(BaseModel):
    decision: str = Field(..., description="FRAUD or NORMAL")
    anomaly_score: float = Field(..., description="ML anomaly score (0–1, higher = riskier)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    rule_severity: int = Field(..., description="Rule engine severity: 0=none, 1=warn, 2=critical")
    triggered_rules: list[str] = Field(..., description="List of business rules that fired")
    ml_flagged: bool = Field(..., description="Whether Isolation Forest flagged this txn")
    latency_ms: float = Field(..., description="Inference time in milliseconds")


# ─────────────────────────────────────────────
# 4. Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/score", response_model=RiskResponse, tags=["scoring"])
def score_transaction(txn: TransactionRequest):
    """
    Score a single transaction.

    Returns a risk decision, anomaly score, and details on which
    rules fired and whether the ML model flagged the transaction.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()

    df = pd.DataFrame([txn.model_dump()])
    result = detector.predict(df).iloc[0]

    latency = (time.perf_counter() - t0) * 1000

    score = float(result["anomaly_score"])
    risk_level = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.35 else "LOW"
    rules = [r for r in result["triggered_rules"].split("; ") if r and r != "none"]

    return RiskResponse(
        decision=result["decision"],
        anomaly_score=round(score, 4),
        risk_level=risk_level,
        rule_severity=int(result["rule_severity"]),
        triggered_rules=rules,
        ml_flagged=bool(result["ml_flag"]),
        latency_ms=round(latency, 2),
    )


@app.post("/score/batch", response_model=list[RiskResponse], tags=["scoring"])
def score_batch(txns: list[TransactionRequest]):
    """
    Score up to 500 transactions in one call.
    Returns results in the same order as input.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    if len(txns) > 500:
        raise HTTPException(status_code=400, detail="Max 500 transactions per batch")

    t0 = time.perf_counter()
    df = pd.DataFrame([t.model_dump() for t in txns])
    results = detector.predict(df)
    latency = (time.perf_counter() - t0) * 1000
    per_txn = round(latency / len(txns), 2)

    out = []
    for _, row in results.iterrows():
        score = float(row["anomaly_score"])
        risk_level = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.35 else "LOW"
        rules = [r for r in row["triggered_rules"].split("; ") if r and r != "none"]
        out.append(RiskResponse(
            decision=row["decision"],
            anomaly_score=round(score, 4),
            risk_level=risk_level,
            rule_severity=int(row["rule_severity"]),
            triggered_rules=rules,
            ml_flagged=bool(row["ml_flag"]),
            latency_ms=per_txn,
        ))
    return out