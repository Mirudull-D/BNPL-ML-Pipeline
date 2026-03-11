import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.feature_engineering import ApplicationData
from pipeline.decision_engine import DecisionEngine

engine = DecisionEngine()
app = ApplicationData(
    applicant_id='G_PAY_STAGE2_TEST',
    age=19,
    annual_income=10000,
    monthly_debt_obligations=5000,
    employment_years=1, # FRONTEND HARDCODES THIS
    employment_status='student',
    order_amount=5000,
    merchant_category='travel',
    installment_plan=12,
    credit_score=None,
    device_age_days=100,
    session_duration_seconds=120,
    email_domain_age_days=400,
    previous_bnpl_orders=0,
    previous_bnpl_defaults=0,
    num_open_accounts=None,
    num_delinquencies_2yr=None,
    credit_utilization=None,
    num_hard_inquiries_6mo=None,
    oldest_account_years=None,
    total_credit_limit=None,
    total_revolving_balance=None,
    num_public_records=None
)

res = engine.evaluate(app, velocity_data={"applications_device_1hr": 0}, simulate_bureau=True)
print(json.dumps(res, indent=2))
