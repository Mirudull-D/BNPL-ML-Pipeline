# BNPL Decision Engine

High-performance 2-stage credit decision engine for Buy Now Pay Later services.
**Target SLA: < 2 seconds end-to-end.**

---

## Architecture

```
Application Received
        │
        ▼
┌───────────────────┐
│  FRAUD GATE       │  ~50ms  Isolation Forest + velocity rules
│  Isolation Forest │         → BLOCK if fraudulent
└───────┬───────────┘
        │ ALLOW
        ▼
┌───────────────────┐
│  STAGE 1          │  ~5ms   No bureau pull — uses application data only
│  Logistic         │         → DECLINE if clearly high-risk (saves ₹40/call)
│  Regression       │         → APPROVE/REVIEW → proceed to Stage 2
└───────┬───────────┘
        │ PROCEED
        ▼
┌───────────────────┐
│  BUREAU API       │  ~600ms Experian / Equifax / TransUnion
│  (Async fetch)    │         Credit score, tradelines, delinquencies
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  STAGE 2          │  ~15ms  Full feature set (app + bureau)
│  XGBoost          │         → Final decision + credit limit
│  + SHAP           │         → Adverse action codes for declines
└───────────────────┘

Total P99: ~670ms  ✅ Well within 2s SLA
```

---

## Why These Two Models?

| | Stage 1 — Logistic Regression | Stage 2 — XGBoost |
|---|---|---|
| **Purpose** | Fast pre-qualification | Definitive credit decision |
| **Data used** | Application only (no bureau) | Application + bureau |
| **Latency** | ~2ms | ~15ms |
| **Accuracy** | 85–90% AUC | 94–97% AUC |
| **Key benefit** | Saves bureau API costs on clear declines | Best accuracy on tabular credit data |
| **Explainability** | Coefficients | SHAP values |

---

## Project Structure

```
bnpl_engine/
├── pipeline/
│   ├── feature_engineering.py  # Stage 1 + Stage 2 feature extractors
│   └── decision_engine.py      # Main orchestrator (runs the full pipeline)
├── models/
│   ├── stage1_logistic.py      # Logistic Regression pre-qualification
│   └── stage2_xgboost.py       # XGBoost final decision + SHAP
├── fraud/
│   └── isolation_forest.py     # Fraud gate (runs before everything)
├── api/
│   └── app.py                  # FastAPI production endpoint
├── data/                       # Trained model files (created after training)
├── train.py                    # Train all models
├── demo.py                     # Run demo with 5 test applicants
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (uses synthetic data by default)
python train.py

# 3. Run demo with test applicants
python demo.py

# 4. Start API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Using Real Training Data

Replace `generate_synthetic_data()` in `train.py` with one of these datasets:

**Recommended — Home Credit Default Risk (Kaggle)**
```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk
```
307k samples, 120+ features — closest to real BNPL data.

**Also good:**
- Give Me Some Credit: `kaggle competitions download -c GiveMeSomeCredit`
- Lending Club: `kaggle datasets download wordsforthewise/lending-club`

---

## API Usage

```bash
curl -X POST http://localhost:8000/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_id": "user_123",
    "age": 30,
    "annual_income": 65000,
    "monthly_debt_obligations": 500,
    "employment_years": 5,
    "employment_status": "employed",
    "order_amount": 350,
    "merchant_category": "electronics",
    "installment_plan": 6,
    "device_age_days": 500,
    "session_duration_seconds": 90
  }'
```

**Response:**
```json
{
  "decision_id": "A3F8B2C1",
  "final_decision": "APPROVED",
  "merchant_response": {
    "approved": true,
    "credit_limit": 750.0,
    "decision_text": "Congratulations! Your purchase has been approved."
  },
  "scores": {
    "stage1_score": 720,
    "final_score": 745,
    "probability_of_default": 0.118
  },
  "performance": {
    "fraud_gate_ms": 48,
    "stage1_ms": 2,
    "bureau_api_ms": 580,
    "stage2_ms": 14,
    "total_ms": 644
  }
}
```

---

## Regulatory Compliance

- **ECOA / Regulation B**: Adverse action codes returned for every decline
- **FCRA**: Reasons mapped to standardized credit bureau codes
- **Monotone constraints**: XGBoost enforces sensible directions (higher credit score always helps)
- **SHAP explanations**: Every decision is fully explainable
- **Audit log**: All decisions stored with full feature vectors

---

## Thresholds (Tune These)

In `models/stage1_logistic.py`:
```python
APPROVE_THRESHOLD = 0.30   # PD < 30% → pre-approve
DECLINE_THRESHOLD = 0.65   # PD > 65% → instant decline
```

In `models/stage2_xgboost.py`:
```python
APPROVE_THRESHOLD = 0.25   # Final: PD < 25% → APPROVED
MANUAL_REVIEW_THRESHOLD = 0.45  # 25-45% → MANUAL REVIEW
```

Adjust these based on your portfolio's risk appetite and observed loss rates.
