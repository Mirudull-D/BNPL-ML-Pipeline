import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.feature_engineering import ApplicationData
from pipeline.decision_engine import DecisionEngine

engine = DecisionEngine()
app = ApplicationData(
    applicant_id='G_PAY_12345',
    age=19,
    annual_income=22000,
    monthly_debt_obligations=2000,
    employment_years=1,
    employment_status='employed',
    order_amount=1000,
    merchant_category='electronics',
    installment_plan=6,
    credit_score=780,
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
result = engine.evaluate(app, velocity_data={"applications_device_1hr": 0}, simulate_bureau=True)

print("\n--- OUTPUT START ---")
print(f"Decision: {result['final_decision']}")
print(f"Reason: {result['final_reason']}")
print(f"Limit: {result['merchant_response']['credit_limit']}")

print("\nStage 1 Result:")
if result['scores']['stage1_score']:
    print(f"  Score: {result['scores']['stage1_score']}")
if result['adverse_action_codes']:
    for code in result['adverse_action_codes']:
        print(f"  Adverse: {code.get('description', '')} {code.get('reason', '')}")

print("\nRisk Factors:")
for rf in result['risk_factors']:
    print(f"  {rf['feature']}: {rf['impact']}")

