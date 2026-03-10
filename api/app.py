"""
BNPL Decision Engine — FastAPI Production Endpoint
─────────────────────────────────────────────────────
Run with: uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints:
  POST /v1/decisions          → Run full decision pipeline
  GET  /v1/decisions/{id}     → Retrieve past decision
  GET  /health                → Health check + model status
  GET  /metrics               → Performance metrics
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_engineering import ApplicationData
from pipeline.decision_engine import DecisionEngine


# ── Pydantic Request/Response Models ──────────────────────────────────────────

class CreditApplicationRequest(BaseModel):
    """API request schema — validates all inputs"""

    # Required fields
    applicant_id: str = Field(..., description="Unique applicant identifier")
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    monthly_debt_obligations: float = Field(..., ge=0)
    employment_years: float = Field(..., ge=0)
    employment_status: str = Field(..., pattern="^(employed|self_employed|student|unemployed)$")
    order_amount: float = Field(..., gt=0, le=10000)
    merchant_category: str = Field(..., pattern="^(electronics|fashion|grocery|luxury|travel)$")
    installment_plan: int = Field(..., pattern="^(3|6|12)$")

    # Optional — device/behavioral signals
    device_age_days: Optional[int] = Field(None, ge=0)
    session_duration_seconds: Optional[int] = Field(None, ge=0)
    email_domain_age_days: Optional[int] = Field(None, ge=0)
    previous_bnpl_orders: Optional[int] = Field(0, ge=0)
    previous_bnpl_defaults: Optional[int] = Field(0, ge=0)

    # Optional — velocity data (from Redis in production)
    velocity_applications_device_1hr: Optional[int] = Field(0, ge=0)
    velocity_applications_email_24hr: Optional[int] = Field(0, ge=0)
    velocity_declines_device_7d: Optional[int] = Field(0, ge=0)

    @validator("installment_plan", pre=True)
    def validate_plan(cls, v):
        if v not in [3, 6, 12]:
            raise ValueError("installment_plan must be 3, 6, or 12")
        return v


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BNPL Decision Engine",
    description="High-performance 2-stage credit decision engine (< 2s SLA)",
    version="1.0.0",
)

# Initialize engine once at startup (not per-request)
engine = DecisionEngine()

# Simple in-memory audit log (use PostgreSQL in production)
decision_log: Dict[str, dict] = {}


@app.on_event("startup")
async def startup():
    """Load models into memory on startup"""
    print("Loading models...")
    # Models are lazy-loaded on first prediction
    # In production, pre-warm here:
    # engine.stage1_model._load_model()
    # engine.stage2_model._load_model()
    print("BNPL Decision Engine ready.")


@app.post("/v1/decisions", summary="Evaluate a credit application")
async def create_decision(request: CreditApplicationRequest) -> JSONResponse:
    """
    Run the 2-stage BNPL credit decision pipeline.
    
    Returns a decision in under 2 seconds with:
    - APPROVED / DECLINED / MANUAL_REVIEW / BLOCKED
    - Approved credit limit
    - Adverse action codes (ECOA/FCRA compliant)
    - Risk factors (from SHAP)
    - Full latency breakdown
    """
    # Build ApplicationData from request
    app_data = ApplicationData(
        applicant_id=request.applicant_id,
        age=request.age,
        annual_income=request.annual_income,
        monthly_debt_obligations=request.monthly_debt_obligations,
        employment_years=request.employment_years,
        employment_status=request.employment_status,
        order_amount=request.order_amount,
        merchant_category=request.merchant_category,
        installment_plan=request.installment_plan,
        device_age_days=request.device_age_days,
        session_duration_seconds=request.session_duration_seconds,
        email_domain_age_days=request.email_domain_age_days,
        previous_bnpl_orders=request.previous_bnpl_orders or 0,
        previous_bnpl_defaults=request.previous_bnpl_defaults or 0,
    )

    velocity_data = {
        "applications_device_1hr": request.velocity_applications_device_1hr or 0,
        "applications_email_24hr": request.velocity_applications_email_24hr or 0,
        "declines_device_7d": request.velocity_declines_device_7d or 0,
    }

    # Run decision pipeline
    result = engine.evaluate(app_data, velocity_data=velocity_data)

    # Audit log (async write to DB in production)
    decision_log[result["decision_id"]] = result

    return JSONResponse(content=result, status_code=200)


@app.get("/v1/decisions/{decision_id}", summary="Retrieve a past decision")
async def get_decision(decision_id: str):
    """Retrieve a past decision by ID (for auditing and adverse action letters)"""
    if decision_id not in decision_log:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return decision_log[decision_id]


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "models": {
            "fraud_detector": engine.fraud_detector.is_trained,
            "stage1_logistic": engine.stage1_model.is_trained,
            "stage2_xgboost": engine.stage2_model.is_trained,
        },
        "decisions_processed": len(decision_log),
    }


@app.get("/metrics", summary="Performance metrics")
async def metrics():
    """Return P50/P95/P99 latency metrics"""
    if not decision_log:
        return {"message": "No decisions processed yet"}

    latencies = [d["performance"]["total_ms"] for d in decision_log.values()]
    latencies.sort()
    n = len(latencies)

    return {
        "total_decisions": n,
        "approval_rate": sum(1 for d in decision_log.values() if d["final_decision"] == "APPROVED") / n,
        "latency_ms": {
            "p50": latencies[int(n * 0.50)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)] if n >= 100 else "need 100+ samples",
            "max": max(latencies),
        },
        "sla_compliance": sum(1 for d in decision_log.values() if d["within_2s_sla"]) / n,
    }
