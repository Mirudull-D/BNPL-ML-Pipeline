"""
Stage 1: Logistic Regression Pre-Qualification Model
─────────────────────────────────────────────────────
Purpose  : Fast, cheap first gate. No bureau pull required.
Input    : Application-only features (income, employment, order, device signals)
Output   : {decision: APPROVE|DECLINE|REVIEW, probability: float, confidence: float}
Latency  : ~2ms inference
"""

import numpy as np
import joblib
import os
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report


class Stage1Model:
    """
    Logistic Regression with StandardScaler.
    
    Why Logistic Regression here:
      - Sub-millisecond inference (critical before bureau pull)
      - Fully explainable coefficients
      - Well-calibrated probabilities out of the box
      - Easy to satisfy adverse action requirements
      - No risk of overfitting on application-only features
    """

    # Decision thresholds — tuned for BNPL risk appetite
    # Adjust APPROVE_THRESHOLD up to be more conservative,
    # down to approve more applications (higher loss risk)
    APPROVE_THRESHOLD = 0.30   # PD < 30% → fast pre-approve, go to Stage 2
    DECLINE_THRESHOLD = 0.65   # PD > 65% → instant decline, no bureau pull
    # Between 30–65% → REVIEW: pull bureau and run Stage 2

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,                    # Moderate regularization (L2)
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",  # Handle class imbalance (few defaults)
                random_state=42,
            ))
        ])
        self.feature_names = None
        self.is_trained = False
        self.model_path = os.path.join(os.path.dirname(__file__), "../data/stage1_model.joblib")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: list):
        """
        Train Stage 1 model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Labels — 1 = defaulted, 0 = repaid
            X_val:   Validation features
            y_val:   Validation labels
            feature_names: List of feature names (for explainability)
        """
        print("=" * 60)
        print("STAGE 1 — Logistic Regression Training")
        print("=" * 60)
        print(f"Training samples : {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Default rate (train): {y_train.mean():.2%}")
        print(f"Features: {len(feature_names)}")
        print()

        self.feature_names = feature_names
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        val_probs = self.pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"Validation AUC-ROC: {auc:.4f}")

        # Show top predictive features (by coefficient magnitude)
        self._print_top_features(n=10)

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, self.model_path)
        print(f"\nModel saved to {self.model_path}")

        return auc

    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Run Stage 1 inference.
        
        Returns:
            {
              "decision": "APPROVE" | "DECLINE" | "REVIEW",
              "probability_of_default": 0.18,
              "confidence": "HIGH" | "LOW",
              "proceed_to_stage2": True | False,
              "stage1_score": 820,   ← scaled 300-850 like a credit score
              "decline_reasons": []  ← populated if DECLINE
            }
        """
        if not self.is_trained:
            self._load_model()

        X = self._dict_to_array(features)
        prob_default = self.pipeline.predict_proba(X)[0, 1]

        decision, proceed = self._apply_thresholds(prob_default)
        score = self._prob_to_score(prob_default)
        confidence = "HIGH" if (prob_default < 0.2 or prob_default > 0.7) else "LOW"
        decline_reasons = self._get_decline_reasons(features, prob_default) if decision == "DECLINE" else []

        return {
            "decision": decision,
            "probability_of_default": round(float(prob_default), 4),
            "stage1_score": score,
            "confidence": confidence,
            "proceed_to_stage2": proceed,
            "decline_reasons": decline_reasons,
        }

    def _apply_thresholds(self, prob: float) -> Tuple[str, bool]:
        if prob < self.APPROVE_THRESHOLD:
            return "APPROVE", True     # Low risk → pre-approve, confirm in Stage 2
        elif prob > self.DECLINE_THRESHOLD:
            return "DECLINE", False    # High risk → instant decline
        else:
            return "REVIEW", True      # Uncertain → pull bureau for Stage 2

    def _prob_to_score(self, prob: float) -> int:
        """Convert probability of default to a 300–850 score (like FICO)"""
        # Higher score = lower risk (inverted from PD)
        score = int(850 - (prob * 550))
        return max(300, min(850, score))

    def _get_decline_reasons(self, features: Dict, prob: float) -> list:
        """
        Generate adverse action codes for ECOA/FCRA compliance.
        These must be returned to declined applicants by law.
        """
        reasons = []

        if features.get("debt_to_income_ratio", 0) > 0.5:
            reasons.append({
                "code": "AA-01",
                "description": "Debt-to-income ratio too high",
                "value": f"{features['debt_to_income_ratio']:.0%}"
            })

        if features.get("employment_risk_score", 0) > 0.5:
            reasons.append({
                "code": "AA-02",
                "description": "Employment status presents elevated risk",
            })

        if features.get("bnpl_default_rate", 0) > 0.2:
            reasons.append({
                "code": "AA-03",
                "description": "Previous BNPL default history",
                "value": f"{features['bnpl_default_rate']:.0%} default rate"
            })

        if features.get("order_to_monthly_income_ratio", 0) > 1.5:
            reasons.append({
                "code": "AA-04",
                "description": "Purchase amount is disproportionate to income",
            })

        if features.get("device_is_new", 0) == 1 and features.get("email_is_new", 0) == 1:
            reasons.append({
                "code": "AA-05",
                "description": "Insufficient account verification history",
            })

        return reasons[:4]  # ECOA requires max 4 adverse action codes

    def _print_top_features(self, n: int = 10):
        """Print top N most predictive features and their direction"""
        coefs = self.pipeline.named_steps["model"].coef_[0]
        feature_importance = list(zip(self.feature_names, coefs))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nTop Predictive Features (Stage 1):")
        print(f"  {'Feature':<40} {'Coefficient':>12} {'Direction'}")
        print("  " + "-" * 65)
        for name, coef in feature_importance[:n]:
            direction = "↑ INCREASES risk" if coef > 0 else "↓ DECREASES risk"
            print(f"  {name:<40} {coef:>+12.4f}   {direction}")

    def _dict_to_array(self, features: Dict) -> np.ndarray:
        arr = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        return arr

    def _load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.pipeline = data["pipeline"]
            self.feature_names = data["feature_names"]
            self.is_trained = True
        else:
            raise RuntimeError("Stage 1 model not trained yet. Run train.py first.")
