"""
BNPL Decision Engine — Main Orchestrator
─────────────────────────────────────────
Runs the full 2-stage pipeline and manages latency budget.

Stage 0 → Fraud Gate         (~50ms)  → BLOCK if fraudulent
Stage 1 → Logistic Regression (~5ms)  → DECLINE instantly if high-risk
                                         (no bureau pull = saves $0.30–0.80/call)
Stage 2 → XGBoost + SHAP     (~15ms)  → Final decision with credit limit
Total P99 target: < 2000ms (bureau API is ~500-800ms of that budget)
"""

import time
import uuid
from typing import Dict, Optional
from dataclasses import asdict

from pipeline.feature_engineering import ApplicationData, Stage1FeatureExtractor, Stage2FeatureExtractor
from models.stage1_logistic import Stage1Model
from models.stage2_xgboost import Stage2Model
from fraud.isolation_forest import FraudDetector


class DecisionEngine:
    """
    Two-stage BNPL credit decision engine.
    
    Workflow:
    ┌─────────────────────────────────────────────────────┐
    │  Application Received                               │
    │         ↓                                           │
    │  [FRAUD GATE] Isolation Forest + velocity rules     │
    │      ↓ BLOCK              ↓ ALLOW/CHALLENGE         │
    │   Reject             [STAGE 1] Logistic Regression  │
    │                 ↙ DECLINE    ↓ APPROVE    ↘ REVIEW  │
    │              Reject     Pull Bureau     Pull Bureau  │
    │                         ↓                  ↓        │
    │                    [STAGE 2] XGBoost + SHAP         │
    │                         ↓                           │
    │                Final Decision + Credit Limit        │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.fraud_detector = FraudDetector()
        self.stage1_model = Stage1Model()
        self.stage2_model = Stage2Model()
        self.stage1_extractor = Stage1FeatureExtractor()
        self.stage2_extractor = Stage2FeatureExtractor()

    def evaluate(self,
                 app: ApplicationData,
                 velocity_data: Optional[Dict] = None,
                 simulate_bureau: bool = True) -> Dict:
        """
        Run the full decision pipeline.
        
        Args:
            app: Application data
            velocity_data: Real-time velocity counts from Redis
            simulate_bureau: If True, simulate bureau API (for testing)
                            In production, this calls real bureau APIs

        Returns:
            Complete decision object with timing breakdown
        """
        pipeline_start = time.perf_counter()
        timings = {}
        decision_id = str(uuid.uuid4())[:8].upper()

        print(f"\n{'='*60}")
        print(f"Decision ID: {decision_id} | Applicant: {app.applicant_id}")
        print(f"{'='*60}")

        # ────────────────────────────────────────────────────────────
        # STAGE 0: FRAUD GATE
        # ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        stage1_features = self.stage1_extractor.extract(app)
        fraud_result = self.fraud_detector.check(stage1_features, velocity_data)
        timings["fraud_gate_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        print(f"[FRAUD GATE]    score={fraud_result['fraud_score']:.3f}  "
              f"action={fraud_result['action']}  ({timings['fraud_gate_ms']}ms)")

        if fraud_result["action"] == "BLOCK":
            total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            return self._build_response(
                decision_id=decision_id,
                app=app,
                final_decision="BLOCKED",
                final_reason="Application blocked by fraud detection",
                fraud_result=fraud_result,
                stage1_result=None,
                stage2_result=None,
                timings=timings,
                total_ms=total_ms,
            )

        # ────────────────────────────────────────────────────────────
        # STAGE 1: LOGISTIC REGRESSION PRE-QUALIFICATION
        # ────────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        stage1_result = self.stage1_model.predict(stage1_features)
        timings["stage1_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        print(f"[STAGE 1]       pd={stage1_result['probability_of_default']:.3f}  "
              f"decision={stage1_result['decision']}  "
              f"score={stage1_result['stage1_score']}  ({timings['stage1_ms']}ms)")

        # Hard decline from Stage 1 — no bureau pull needed
        if not stage1_result["proceed_to_stage2"]:
            total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            print(f"[PIPELINE]      EARLY DECLINE (no bureau pull) — saved ~$0.50")
            return self._build_response(
                decision_id=decision_id,
                app=app,
                final_decision="DECLINED",
                final_reason="Pre-qualification failed (Stage 1)",
                fraud_result=fraud_result,
                stage1_result=stage1_result,
                stage2_result=None,
                timings=timings,
                total_ms=total_ms,
            )

        # ────────────────────────────────────────────────────────────
        # BUREAU DATA FETCH (simulated; in production: real API call)
        # ────────────────────────────────────────────────────────────
        t2 = time.perf_counter()
        if simulate_bureau:
            app = self._simulate_bureau_data(app)
            # In production, this would be:
            # bureau_data = await experian_client.fetch(app.applicant_id)
            # app = merge_bureau_data(app, bureau_data)
        timings["bureau_api_ms"] = round((time.perf_counter() - t2) * 1000, 2)
        print(f"[BUREAU API]    credit_score={app.credit_score}  "
              f"utilization={app.credit_utilization:.0%}  ({timings['bureau_api_ms']}ms)")

        # ────────────────────────────────────────────────────────────
        # STAGE 2: XGBOOST FULL CREDIT DECISION
        # ────────────────────────────────────────────────────────────
        t3 = time.perf_counter()
        stage2_features = self.stage2_extractor.extract(app)
        stage2_result = self.stage2_model.predict(stage2_features, annual_income=app.annual_income)
        timings["stage2_ms"] = round((time.perf_counter() - t3) * 1000, 2)

        print(f"[STAGE 2]       pd={stage2_result['probability_of_default']:.3f}  "
              f"decision={stage2_result['decision']}  "
              f"score={stage2_result['final_score']}  ({timings['stage2_ms']}ms)")

        # ────────────────────────────────────────────────────────────
        # FINAL RESPONSE
        # ────────────────────────────────────────────────────────────
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        within_sla = total_ms < 2000

        print(f"\n[TOTAL]         {total_ms}ms  ({'✅ WITHIN SLA' if within_sla else '❌ SLA BREACH'})")
        print(f"{'='*60}")

        return self._build_response(
            decision_id=decision_id,
            app=app,
            final_decision=stage2_result["decision"],
            final_reason=f"XGBoost credit model — PD: {stage2_result['probability_of_default']:.1%}",
            fraud_result=fraud_result,
            stage1_result=stage1_result,
            stage2_result=stage2_result,
            timings=timings,
            total_ms=total_ms,
        )

    def _build_response(self, decision_id, app, final_decision, final_reason,
                        fraud_result, stage1_result, stage2_result, timings, total_ms) -> Dict:
        """Build the structured API response"""
        response = {
            "decision_id": decision_id,
            "applicant_id": app.applicant_id,
            "final_decision": final_decision,
            "final_reason": final_reason,
            "within_2s_sla": total_ms < 2000,

            # What the merchant needs
            "merchant_response": {
                "approved": final_decision == "APPROVED",
                "credit_limit": stage2_result["approved_credit_limit"] if stage2_result else 0,
                "decision_text": self._get_decision_text(final_decision),
            },

            # Scoring details
            "scores": {
                "stage1_score": stage1_result["stage1_score"] if stage1_result else None,
                "final_score": stage2_result["final_score"] if stage2_result else None,
                "probability_of_default": stage2_result["probability_of_default"] if stage2_result else None,
            },

            # Compliance — required by ECOA/FCRA for declines
            "adverse_action_codes": (
                stage2_result.get("adverse_action_codes", []) if stage2_result
                else stage1_result.get("decline_reasons", []) if stage1_result
                else []
            ),

            # Risk factors from SHAP
            "risk_factors": stage2_result.get("top_risk_factors", []) if stage2_result else [],

            # Fraud
            "fraud": {
                "score": fraud_result["fraud_score"],
                "action": fraud_result["action"],
                "signals": fraud_result["fraud_signals"],
            },

            # Pipeline performance
            "performance": {
                **timings,
                "total_ms": total_ms,
            },
        }
        return response

    def _get_decision_text(self, decision: str) -> str:
        texts = {
            "APPROVED": "Congratulations! Your purchase has been approved.",
            "DECLINED": "We're unable to approve this purchase at this time.",
            "MANUAL_REVIEW": "Your application is under review. We'll update you shortly.",
            "BLOCKED": "We're unable to process this application.",
        }
        return texts.get(decision, "Application processed.")

    def _simulate_bureau_data(self, app: ApplicationData) -> ApplicationData:
        """
        Simulate bureau API response for testing.
        In production, replace with real Experian/Equifax API call.
        """
        import random
        rng = random.Random(hash(app.applicant_id))

        # Generate realistic bureau data correlated with income/employment
        base_score = 580 + (app.annual_income / 1000) + (app.employment_years * 5)
        base_score = max(300, min(850, base_score + rng.gauss(0, 50)))

        app.credit_score = round(base_score)
        app.num_open_accounts = rng.randint(2, 15)
        app.num_delinquencies_2yr = rng.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
        app.credit_utilization = round(rng.uniform(0.05, 0.90), 2)
        app.num_hard_inquiries_6mo = rng.choices([0, 1, 2, 3, 4], weights=[40, 30, 15, 10, 5])[0]
        app.oldest_account_years = round(rng.uniform(0.5, 20), 1)
        app.total_credit_limit = round(app.annual_income * rng.uniform(0.2, 1.5), -2)
        app.total_revolving_balance = round(app.total_credit_limit * app.credit_utilization, -1)
        app.num_public_records = rng.choices([0, 1, 2], weights=[88, 9, 3])[0]

        return app
