"""
Stage 2: XGBoost Full Credit Decision Model
─────────────────────────────────────────────
Purpose  : Definitive credit decision using full bureau + application data
Input    : All features (Stage 1 features + bureau data)
Output   : {decision, probability, credit_limit, shap_explanations}
Latency  : ~15ms inference
"""

import numpy as np
import joblib
import os
from typing import Dict, List, Optional
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import shap


class Stage2Model:
    """
    XGBoost Credit Scoring Model.
    
    Why XGBoost here:
      - Best accuracy on tabular credit data (94-97% AUC)
      - Handles missing bureau fields natively (no imputation needed)
      - Built-in feature importance
      - Monotone constraints: enforce that higher credit score = lower risk
        (regulatory requirement in many jurisdictions)
      - SHAP explanations for adverse action codes
      - ~15ms inference (well within 2s budget)
    """

    # Final decision thresholds
    APPROVE_THRESHOLD = 0.25    # PD < 25% → APPROVED
    MANUAL_REVIEW_THRESHOLD = 0.45  # 25–45% → MANUAL REVIEW
    DECLINE_THRESHOLD = 0.45    # PD > 45% → DECLINED

    # Credit limit rules (% of monthly income × plan multiplier)
    CREDIT_LIMIT_RULES = {
        "APPROVED": {"base_pct": 1.5, "max_absolute": 5000},
        "REDUCED_AMOUNT": {"base_pct": 0.75, "max_absolute": 1500},
    }

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,        # Prevents overfitting on small groups
            scale_pos_weight=10,        # ~10:1 class imbalance in credit data
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            n_jobs=-1,                  # Use all CPU cores for training
            tree_method="hist",         # Fastest training method

            # ─── Monotone Constraints ───────────────────────────────────────
            # Ensures model behaves sensibly for regulators:
            # credit_score higher → risk lower (must be monotone decreasing)
            # delinquencies higher → risk higher (must be monotone increasing)
            # These are set during training via monotone_constraints param
        )
        self.explainer = None
        self.feature_names = None
        self.is_trained = False
        self.model_path = os.path.join(os.path.dirname(__file__), "../data/stage2_model.joblib")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: List[str],
              monotone_constraints: Optional[Dict[str, int]] = None):
        """
        Train Stage 2 XGBoost model.
        
        Args:
            X_train: Full feature matrix (app + bureau features)
            y_train: Labels — 1 = defaulted, 0 = repaid
            X_val:   Validation features
            y_val:   Validation labels
            feature_names: Feature names in order
            monotone_constraints: Dict of {feature_name: direction}
                direction = +1 (higher = more risk) or -1 (higher = less risk)
                Example: {"credit_score_normalized": -1, "num_delinquencies_2yr": +1}
        """
        print("=" * 60)
        print("STAGE 2 — XGBoost Training")
        print("=" * 60)
        print(f"Training samples  : {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Default rate (train): {y_train.mean():.2%}")
        print(f"Total features: {len(feature_names)}")

        self.feature_names = feature_names

        # Build monotone constraints vector if provided
        if monotone_constraints:
            constraint_vec = tuple(
                monotone_constraints.get(f, 0) for f in feature_names
            )
            self.model.set_params(monotone_constraints=constraint_vec)
            print(f"Monotone constraints applied: {sum(v != 0 for v in constraint_vec)} features")

        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        self.is_trained = True

        # Evaluate
        val_probs = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"\nFinal Validation AUC-ROC: {auc:.4f}")

        # Build SHAP explainer (TreeExplainer is fast, ~1ms per sample)
        print("\nBuilding SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("SHAP explainer ready.")

        # Top features
        self._print_top_features(n=15)

        # Save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
        }, self.model_path)
        print(f"\nModel saved to {self.model_path}")

        return auc

    def predict(self, features: Dict[str, float], annual_income: float = 0) -> Dict:
        """
        Run Stage 2 inference with full explanations.
        
        Returns:
            {
              "decision": "APPROVED" | "DECLINED" | "MANUAL_REVIEW",
              "probability_of_default": 0.12,
              "final_score": 760,
              "approved_credit_limit": 1200.0,
              "approved_installment_plan": 6,
              "monthly_payment": 200.0,
              "top_risk_factors": [...],
              "top_protective_factors": [...],
              "adverse_action_codes": [...],
              "shap_values": {...}
            }
        """
        if not self.is_trained:
            self._load_model()

        X = self._dict_to_array(features)
        prob_default = float(self.model.predict_proba(X)[0, 1])

        decision = self._make_decision(prob_default)
        score = self._prob_to_score(prob_default)
        credit_limit = self._calculate_credit_limit(decision, annual_income, prob_default)
        shap_explanation = self._explain(X, prob_default, decision)

        result = {
            "decision": decision,
            "probability_of_default": round(prob_default, 4),
            "final_score": score,
            "approved_credit_limit": credit_limit,
            "top_risk_factors": shap_explanation["top_risk_factors"],
            "top_protective_factors": shap_explanation["top_protective_factors"],
            "adverse_action_codes": shap_explanation["adverse_action_codes"] if decision == "DECLINED" else [],
            "model": "XGBoost-v1",
        }

        return result

    def _make_decision(self, prob: float) -> str:
        if prob < self.APPROVE_THRESHOLD:
            return "APPROVED"
        elif prob < self.MANUAL_REVIEW_THRESHOLD:
            return "REDUCED_AMOUNT"
        else:
            return "DECLINED"

    def _prob_to_score(self, prob: float) -> int:
        score = int(850 - (prob * 550))
        return max(300, min(850, score))

    def _calculate_credit_limit(self, decision: str, annual_income: float, prob: float) -> float:
        """
        Calculate approved credit limit based on income and risk.
        Higher risk = lower limit, even within approved band.
        """
        if decision == "DECLINED":
            return 0.0

        monthly_income = annual_income / 12
        rules = self.CREDIT_LIMIT_RULES.get(decision, {"base_pct": 0.5, "max_absolute": 500})

        # Risk-adjusted limit: reduce as probability increases
        risk_adjustment = 1.0 - (prob / self.APPROVE_THRESHOLD) * 0.3
        raw_limit = monthly_income * rules["base_pct"] * risk_adjustment
        limit = min(raw_limit, rules["max_absolute"])

        # Round to nearest $50
        return round(max(50.0, limit) / 50) * 50

    def _explain(self, X: np.ndarray, prob: float, decision: str) -> Dict:
        """Generate SHAP explanations for transparency and adverse action codes"""
        if self.explainer is None:
            return {"top_risk_factors": [], "top_protective_factors": [], "adverse_action_codes": []}

        shap_values = self.explainer.shap_values(X)[0]  # Single sample
        feature_impacts = list(zip(self.feature_names, shap_values))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)

        # Positive SHAP = increases default risk
        risk_factors = [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in feature_impacts if v > 0
        ][:5]

        # Negative SHAP = decreases default risk (protective)
        protective_factors = [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in reversed(feature_impacts) if v < 0
        ][:5]

        # Adverse action codes for declined applications (ECOA requirement)
        adverse_codes = []
        if decision == "DECLINED":
            for factor in risk_factors[:4]:
                code = self._map_feature_to_adverse_code(factor["feature"])
                if code:
                    adverse_codes.append(code)

        return {
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "adverse_action_codes": adverse_codes,
        }

    def _map_feature_to_adverse_code(self, feature_name: str) -> Optional[Dict]:
        """Map feature names to standardized adverse action descriptions"""
        ADVERSE_CODES = {
            "num_delinquencies_2yr":        {"code": "AA-01", "reason": "Delinquent past or present credit obligations"},
            "credit_utilization":           {"code": "AA-02", "reason": "Level of delinquency on accounts"},
            "num_hard_inquiries_6mo":        {"code": "AA-03", "reason": "Too many inquiries in last 12 months"},
            "debt_to_income_ratio":          {"code": "AA-04", "reason": "Income insufficient for debt obligations"},
            "oldest_account_years":          {"code": "AA-05", "reason": "Length of credit history too short"},
            "num_open_accounts":             {"code": "AA-06", "reason": "Too few accounts currently paid as agreed"},
            "num_public_records":            {"code": "AA-07", "reason": "Derogatory public record or collection filed"},
            "employment_risk_score":         {"code": "AA-08", "reason": "Unable to verify employment"},
            "order_to_monthly_income_ratio": {"code": "AA-09", "reason": "Purchase amount exceeds income threshold"},
            "bnpl_default_rate":             {"code": "AA-10", "reason": "Prior BNPL payment performance"},
        }
        return ADVERSE_CODES.get(feature_name)

    def _print_top_features(self, n: int = 15):
        importance = self.model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1], reverse=True
        )
        print(f"\nTop {n} Feature Importances (Stage 2 XGBoost):")
        print(f"  {'Feature':<45} {'Importance':>12}")
        print("  " + "-" * 60)
        for name, imp in feature_importance[:n]:
            bar = "█" * int(imp * 200)
            print(f"  {name:<45} {imp:>12.4f}  {bar}")

    def _dict_to_array(self, features: Dict) -> np.ndarray:
        return np.array([[features.get(f, 0.0) for f in self.feature_names]])

    def _load_model(self):
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.explainer = data["explainer"]
            self.feature_names = data["feature_names"]
            self.is_trained = True
        else:
            raise RuntimeError("Stage 2 model not trained. Run train.py first.")
