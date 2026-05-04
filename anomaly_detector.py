"""
Transaction Anomaly Detector
============================
Hybrid approach: Rule-based pre-filter + Isolation Forest (scikit-learn)
Trained on synthetic transaction data.

Usage:
    pip install scikit-learn pandas numpy
    python anomaly_detector.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Synthetic Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_transactions(
    n_normal: int = 2000,
    n_fraud: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic transaction dataset.

    Normal transactions:
      - Amount: $5–$500, log-normal distribution
      - Hour: any, weighted toward business hours
      - Frequency: 1–5 transactions per day per account
      - Same country as account origin

    Anomalous transactions:
      - Large amounts (>$3000)
      - Unusual hours (2–4 AM)
      - High velocity (many transactions in short period)
      - Foreign country mismatch
    """
    rng = np.random.default_rng(random_state)

    def make_normal(n):
        amounts     = np.round(np.exp(rng.normal(4.5, 1.0, n)).clip(5, 500), 2)
        hours       = rng.choice(range(24), n, p=_hour_weights())
        days_since  = rng.integers(0, 30, n)
        freq_1h     = rng.integers(1, 4, n)
        freq_24h    = rng.integers(1, 8, n)
        country_match = rng.choice([0, 1], n, p=[0.05, 0.95])  # mostly home country
        merchant_risk = rng.choice([0, 1, 2], n, p=[0.70, 0.25, 0.05])  # low/med/high
        declined_prev = rng.integers(0, 2, n)
        return amounts, hours, days_since, freq_1h, freq_24h, country_match, merchant_risk, declined_prev

    def make_fraud(n):
        # Mix of anomaly types
        amounts        = np.round(rng.choice([
            rng.uniform(3000, 15000, n),   # large
            rng.uniform(0.01, 1.00, n),    # micro (testing card)
        ], axis=0)[0], 2)
        hours          = rng.choice([2, 3, 4, 23], n)            # late night
        days_since     = rng.integers(0, 5, n)
        freq_1h        = rng.integers(8, 30, n)                  # velocity spike
        freq_24h       = rng.integers(20, 60, n)
        country_match  = rng.choice([0, 1], n, p=[0.80, 0.20])  # mostly foreign
        merchant_risk  = rng.choice([0, 1, 2], n, p=[0.10, 0.30, 0.60])
        declined_prev  = rng.integers(1, 5, n)
        return amounts, hours, days_since, freq_1h, freq_24h, country_match, merchant_risk, declined_prev

    na, nh, nd, nf1, nf24, ncm, nmr, ndp = make_normal(n_normal)
    fa, fh, fd, ff1, ff24, fcm, fmr, fdp = make_fraud(n_fraud)

    df = pd.DataFrame({
        "amount":           np.concatenate([na, fa]),
        "hour":             np.concatenate([nh, fh]),
        "days_since_first": np.concatenate([nd, fd]),
        "txn_freq_1h":      np.concatenate([nf1, ff1]),
        "txn_freq_24h":     np.concatenate([nf24, ff24]),
        "country_match":    np.concatenate([ncm, fcm]),
        "merchant_risk":    np.concatenate([nmr, fmr]),
        "declined_prev":    np.concatenate([ndp, fdp]),
        "is_fraud":         np.array([0]*n_normal + [1]*n_fraud),
    })
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _hour_weights():
    """Business-hour weighted probability distribution over 24 hours."""
    w = np.ones(24) * 0.02
    w[8:20] = 0.05     # business hours
    w[12:14] = 0.07    # lunch peak
    return w / w.sum()


# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features for better anomaly detection."""
    df = df.copy()
    df["log_amount"]       = np.log1p(df["amount"])
    df["is_night"]         = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["velocity_ratio"]   = df["txn_freq_1h"] / (df["txn_freq_24h"].clip(1))
    df["risk_score_raw"]   = (
        df["merchant_risk"] * 2
        + df["declined_prev"]
        + (1 - df["country_match"]) * 3
        + df["is_night"]
    )
    return df


FEATURES = [
    "log_amount", "is_night", "txn_freq_1h", "txn_freq_24h",
    "velocity_ratio", "country_match", "merchant_risk",
    "declined_prev", "risk_score_raw",
]


# ─────────────────────────────────────────────
# 3. Rule-Based Pre-Filter
# ─────────────────────────────────────────────

class RuleBasedFilter:
    """
    Hard business rules that flag obvious anomalies before the ML model.
    Returns a risk flag (0 = normal, 1 = suspicious, 2 = critical).
    """

    RULES = [
        # (description, condition_fn, severity)
        ("Amount > $5,000",          lambda r: r["amount"] > 5000,        2),
        ("Amount < $0.10 (card test)",lambda r: r["amount"] < 0.10,       2),
        ("Night + foreign country",  lambda r: r["is_night"] and not r["country_match"], 2),
        (">10 txns in 1 hour",       lambda r: r["txn_freq_1h"] > 10,     1),
        (">2 previous declines",     lambda r: r["declined_prev"] > 2,    1),
        ("High-risk merchant + night",lambda r: r["merchant_risk"] == 2 and r["is_night"], 1),
    ]

    def evaluate(self, row: dict) -> tuple[int, list[str]]:
        """Returns (max_severity, [triggered_rule_descriptions])."""
        triggered, max_sev = [], 0
        for desc, cond, sev in self.RULES:
            try:
                if cond(row):
                    triggered.append(desc)
                    max_sev = max(max_sev, sev)
            except Exception:
                pass
        return max_sev, triggered


# ─────────────────────────────────────────────
# 4. ML Model (Isolation Forest)
# ─────────────────────────────────────────────

class MLAnomalyDetector:
    """
    Unsupervised Isolation Forest anomaly detector.
    Contamination is set to the expected fraud rate (~5%).
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  IsolationForest(
                n_estimators=200,
                contamination=contamination,
                max_samples="auto",
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self.trained = False

    def fit(self, X: pd.DataFrame):
        # Train only on normal data (unsupervised — we pretend labels are unavailable)
        self.pipeline.fit(X[FEATURES])
        self.trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        raw = self.pipeline.predict(X[FEATURES])   # -1 = anomaly, 1 = normal
        return (raw == -1).astype(int)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns anomaly score (higher = more anomalous)."""
        raw_scores = self.pipeline.decision_function(X[FEATURES])
        return -raw_scores   # flip so higher = more anomalous


# ─────────────────────────────────────────────
# 5. Hybrid Detector
# ─────────────────────────────────────────────

class TransactionAnomalyDetector:
    """
    Combines rule-based hard filters with Isolation Forest ML.
    Decision logic:
      - Critical rule → FRAUD regardless of ML
      - ML anomaly OR suspicious rule → FRAUD
      - Neither → NORMAL
    """

    def __init__(self):
        self.rules = RuleBasedFilter()
        self.ml    = MLAnomalyDetector()

    def fit(self, df: pd.DataFrame):
        df_feat = engineer_features(df)
        self.ml.fit(df_feat)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat   = engineer_features(df)
        ml_preds  = self.ml.predict(df_feat)
        ml_scores = self.ml.score(df_feat)

        results = []
        for i, row in df_feat.iterrows():
            rule_sev, triggered = self.rules.evaluate(row.to_dict())
            ml_flag = int(ml_preds[df_feat.index.get_loc(i)])
            score   = float(ml_scores[df_feat.index.get_loc(i)])

            # Decision logic
            if rule_sev == 2 or (rule_sev >= 1 and ml_flag == 1) or ml_flag == 1:
                decision = "FRAUD"
            else:
                decision = "NORMAL"

            results.append({
                "amount":          row["amount"],
                "hour":            int(row["hour"]),
                "decision":        decision,
                "ml_flag":         ml_flag,
                "rule_severity":   rule_sev,
                "anomaly_score":   round(score, 4),
                "triggered_rules": "; ".join(triggered) if triggered else "none",
                "true_label":      int(row.get("is_fraud", -1)),
            })

        return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 6. Main — Train, Evaluate, Demo
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Transaction Anomaly Detector")
    print("  Isolation Forest + Rule Engine")
    print("=" * 60)

    # --- Generate data ---
    print("\n[1/4] Generating synthetic transaction data...")
    df = generate_synthetic_transactions(n_normal=2000, n_fraud=100)
    print(f"      Total: {len(df):,} transactions | Fraud: {df['is_fraud'].sum()}")

    # --- Train ---
    print("\n[2/4] Training detector (Isolation Forest, 200 trees)...")
    detector = TransactionAnomalyDetector()
    detector.fit(df)
    print("      Training complete.")

    # --- Evaluate ---
    print("\n[3/4] Evaluating on full dataset...")
    results = detector.predict(df)
    y_true  = results["true_label"]
    y_pred  = (results["decision"] == "FRAUD").astype(int)

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {tp/(tp+fp):.2%}  Recall: {tp/(tp+fn):.2%}")

    # --- Live demo on new transactions ---
    print("\n[4/4] Demo — scoring 5 new transactions:\n")
    demo_txns = pd.DataFrame([
        {"amount": 12.50,   "hour": 14, "txn_freq_1h": 2,  "txn_freq_24h": 5,  "country_match": 1, "merchant_risk": 0, "declined_prev": 0, "days_since_first": 365},
        {"amount": 7500.00, "hour":  3, "txn_freq_1h": 15, "txn_freq_24h": 40, "country_match": 0, "merchant_risk": 2, "declined_prev": 3, "days_since_first": 1},
        {"amount": 0.01,    "hour":  2, "txn_freq_1h": 20, "txn_freq_24h": 35, "country_match": 0, "merchant_risk": 2, "declined_prev": 4, "days_since_first": 0},
        {"amount": 89.99,   "hour": 19, "txn_freq_1h": 3,  "txn_freq_24h": 7,  "country_match": 1, "merchant_risk": 1, "declined_prev": 0, "days_since_first": 120},
        {"amount": 250.00,  "hour":  1, "txn_freq_1h": 1,  "txn_freq_24h": 2,  "country_match": 0, "merchant_risk": 0, "declined_prev": 0, "days_since_first": 200},
    ])
    demo_results = detector.predict(demo_txns)

    for _, r in demo_results.iterrows():
        icon = "🚨" if r["decision"] == "FRAUD" else "✅"
        print(f"  {icon} ${r['amount']:>9.2f} @ {int(r['hour']):02d}:00h  →  "
              f"{r['decision']:<6}  score={r['anomaly_score']:.3f}  "
              f"rules=[{r['triggered_rules']}]")

    print("\n" + "=" * 60)
    print("  Done. Export results with: results.to_csv('results.csv')")
    print("=" * 60)

    return detector, results


if __name__ == "__main__":
    detector, results = main()