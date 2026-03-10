"""
Fraud Detection Layer — Runs BEFORE credit scoring
─────────────────────────────────────────────────────
Model  : Isolation Forest (unsupervised anomaly detection)
Purpose: Catch bots, stolen identities, velocity fraud, device farms
Latency: ~5ms
"""

import numpy as np
import joblib
import os
from typing import Dict
from sklearn.ensemble import IsolationForest


class FraudDetector:
    """
    Isolation Forest fraud gate.
    
    Runs before any credit model — rejects obvious fraud before
    wasting bureau API calls or model compute.
    
    Why Isolation Forest:
      - Unsupervised: no labeled fraud data needed to get started
      - Detects novel fraud patterns not seen before
      - Lightweight: ~5ms inference
      - Low false positive rate when contamination is tuned correctly
    """

    # If fraud score > threshold, reject the application
    FRAUD_SCORE_THRESHOLD = 0.65

    # Velocity limits (checked in-memory via Redis in production)
    VELOCITY_RULES = {
        "max_applications_per_device_1hr": 3,
        "max_applications_per_email_24hr": 2,
        "max_applications_per_ip_1hr": 5,
        "max_declined_per_device_7d": 5,
    }

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,  # ~2% of applications expected to be fraud
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False
        self.model_path = os.path.join(os.path.dirname(__file__), "../data/fraud_model.joblib")

    def train(self, X_normal: np.ndarray):
        """
        Train on NORMAL (non-fraud) applications only.
        Isolation Forest learns what normal looks like, then flags outliers.
        """
        print("=" * 60)
        print("FRAUD DETECTOR — Isolation Forest Training")
        print("=" * 60)
        print(f"Training on {len(X_normal):,} normal samples")

        self.model.fit(X_normal)
        self.is_trained = True

        # Test: how many training samples flagged as anomalies?
        preds = self.model.predict(X_normal)
        flagged = (preds == -1).sum()
        print(f"Flagged as anomalies in training set: {flagged} ({flagged/len(X_normal):.1%})")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Fraud model saved to {self.model_path}")

    def check(self, features: Dict[str, float], velocity_data: Dict = None) -> Dict:
        """
        Run fraud checks. Returns fraud assessment.
        
        Args:
            features: Application features dict
            velocity_data: Real-time counts from Redis
                          {"applications_this_hour": 1, "declines_this_week": 0, ...}
        
        Returns:
            {
              "is_fraud": False,
              "fraud_score": 0.12,       ← 0.0 = normal, 1.0 = highly suspicious
              "fraud_signals": [...],     ← list of triggered rules
              "action": "ALLOW" | "BLOCK" | "CHALLENGE"
            }
        """
        fraud_signals = []
        fraud_score = 0.0

        # ── Rule-Based Checks (fastest, run first) ───────────────────────
        velocity_data = velocity_data or {}

        if velocity_data.get("applications_device_1hr", 0) >= self.VELOCITY_RULES["max_applications_per_device_1hr"]:
            fraud_signals.append({"rule": "VELOCITY_DEVICE", "severity": "HIGH", "desc": "Too many applications from same device"})
            fraud_score += 0.4

        if velocity_data.get("applications_email_24hr", 0) >= self.VELOCITY_RULES["max_applications_per_email_24hr"]:
            fraud_signals.append({"rule": "VELOCITY_EMAIL", "severity": "HIGH", "desc": "Too many applications from same email"})
            fraud_score += 0.3

        if velocity_data.get("declines_device_7d", 0) >= self.VELOCITY_RULES["max_declined_per_device_7d"]:
            fraud_signals.append({"rule": "DECLINED_VELOCITY", "severity": "MEDIUM", "desc": "Multiple declines from same device recently"})
            fraud_score += 0.25

        # New device + new email = high risk signal
        if features.get("device_is_new", 0) == 1 and features.get("email_is_new", 0) == 1:
            fraud_signals.append({"rule": "NEW_DEVICE_AND_EMAIL", "severity": "MEDIUM", "desc": "Both device and email are newly created"})
            fraud_score += 0.2

        # Impossible session signals (too fast = bot)
        if features.get("session_duration_seconds", 999) < 8:
            fraud_signals.append({"rule": "BOT_SPEED", "severity": "HIGH", "desc": "Application completed suspiciously fast (bot signal)"})
            fraud_score += 0.35

        # ── ML Anomaly Score ──────────────────────────────────────────────
        if self.is_trained:
            ml_fraud_score = self._ml_score(features)
            fraud_score = min(1.0, fraud_score + ml_fraud_score * 0.3)

            if ml_fraud_score > 0.6:
                fraud_signals.append({
                    "rule": "ML_ANOMALY",
                    "severity": "MEDIUM",
                    "desc": f"Behavioral pattern is anomalous (score: {ml_fraud_score:.2f})"
                })

        # ── Final Decision ────────────────────────────────────────────────
        fraud_score = min(1.0, fraud_score)

        if fraud_score >= self.FRAUD_SCORE_THRESHOLD:
            action = "BLOCK"
        elif fraud_score >= 0.35:
            action = "CHALLENGE"   # Require 2FA / additional verification
        else:
            action = "ALLOW"

        return {
            "is_fraud": action == "BLOCK",
            "fraud_score": round(fraud_score, 4),
            "fraud_signals": fraud_signals,
            "action": action,
        }

    def _ml_score(self, features: Dict) -> float:
        """
        Convert Isolation Forest output to 0–1 fraud score.
        IF returns negative scores for anomalies; we normalize to 0–1.
        """
        if not self.is_trained:
            return 0.0

        # Use a simple feature subset for fraud (no bureau data needed)
        fraud_features = [
            features.get("session_duration_seconds", 0),
            features.get("device_age_days", 0),
            features.get("email_domain_age_days", 0),
            features.get("device_is_new", 0),
            features.get("email_is_new", 0),
            features.get("order_amount_log", 0),
            features.get("order_to_monthly_income_ratio", 0),
        ]

        X = np.array([fraud_features])
        raw_score = self.model.score_samples(X)[0]

        # Normalize: IF scores range roughly -0.5 to 0.5
        # More negative = more anomalous → higher fraud score
        normalized = max(0.0, min(1.0, (-raw_score + 0.3) * 2))
        return normalized

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
