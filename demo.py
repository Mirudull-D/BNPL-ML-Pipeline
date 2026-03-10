"""
Demo Script — Test the BNPL Decision Engine
Run this after training: python demo.py

Tests 5 applicant profiles ranging from excellent to high-risk.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.feature_engineering import ApplicationData
from pipeline.decision_engine import DecisionEngine


def print_result(result: dict):
    dec = result["final_decision"]
    colors = {"APPROVED": "\033[92m", "DECLINED": "\033[91m",
              "MANUAL_REVIEW": "\033[93m", "BLOCKED": "\033[91m"}
    reset = "\033[0m"
    color = colors.get(dec, "")

    print(f"\n  Decision : {color}{dec}{reset}")
    print(f"  Score    : {result['scores'].get('final_score', result['scores'].get('stage1_score', 'N/A'))}")
    print(f"  PD       : {(result['scores'].get('probability_of_default') or 0):.1%}")
    print(f"  Limit    : ₹{result['merchant_response']['credit_limit']:,.0f}")
    print(f"  Latency  : {result['performance']['total_ms']}ms {'✅' if result['within_2s_sla'] else '❌'}")

    if result.get("adverse_action_codes"):
        print(f"  Decline reasons:")
        for code in result["adverse_action_codes"][:3]:
            print(f"    [{code.get('code', 'AA-XX')}] {code.get('reason', code.get('description', ''))}")

    if result.get("risk_factors"):
        print(f"  Top risk factors: {', '.join(f['feature'] for f in result['risk_factors'][:3])}")


def run_demo():
    engine = DecisionEngine()

    test_cases = [
        {
            "name": "Sarah — Excellent Profile",
            "app": ApplicationData(
                applicant_id="SARAH_001", age=34,
                annual_income=95000, monthly_debt_obligations=800,
                employment_years=8, employment_status="employed",
                order_amount=450, merchant_category="electronics",
                installment_plan=6, device_age_days=720,
                session_duration_seconds=95, email_domain_age_days=1800,
                previous_bnpl_orders=5, previous_bnpl_defaults=0,
            ),
        },
        {
            "name": "Marcus — Good Profile",
            "app": ApplicationData(
                applicant_id="MARCUS_002", age=28,
                annual_income=55000, monthly_debt_obligations=600,
                employment_years=3, employment_status="employed",
                order_amount=280, merchant_category="fashion",
                installment_plan=3, device_age_days=400,
                session_duration_seconds=120, email_domain_age_days=600,
                previous_bnpl_orders=2, previous_bnpl_defaults=0,
            ),
        },
        {
            "name": "Aisha — Borderline Profile",
            "app": ApplicationData(
                applicant_id="AISHA_003", age=22,
                annual_income=30000, monthly_debt_obligations=900,
                employment_years=1, employment_status="employed",
                order_amount=400, merchant_category="electronics",
                installment_plan=12, device_age_days=180,
                session_duration_seconds=60, email_domain_age_days=200,
                previous_bnpl_orders=1, previous_bnpl_defaults=1,
            ),
        },
        {
            "name": "Derek — High Risk Profile",
            "app": ApplicationData(
                applicant_id="DEREK_004", age=45,
                annual_income=22000, monthly_debt_obligations=1800,
                employment_years=0.5, employment_status="self_employed",
                order_amount=800, merchant_category="luxury",
                installment_plan=12, device_age_days=30,
                session_duration_seconds=45, email_domain_age_days=15,
                previous_bnpl_orders=4, previous_bnpl_defaults=2,
            ),
        },
        {
            "name": "Bot — Fraud Attempt",
            "app": ApplicationData(
                applicant_id="BOT_005", age=25,
                annual_income=50000, monthly_debt_obligations=400,
                employment_years=3, employment_status="employed",
                order_amount=2000, merchant_category="electronics",
                installment_plan=3, device_age_days=1,
                session_duration_seconds=3,   # Too fast = bot
                email_domain_age_days=2,       # Brand new email
                previous_bnpl_orders=0, previous_bnpl_defaults=0,
            ),
            "velocity": {"applications_device_1hr": 5}  # Velocity fraud
        },
    ]

    print("\n" + "="*60)
    print("  BNPL Decision Engine — Demo")
    print("="*60)

    for case in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Applicant: {case['name']}")
        result = engine.evaluate(
            case["app"],
            velocity_data=case.get("velocity", {}),
            simulate_bureau=True
        )
        print_result(result)

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print("="*60)


if __name__ == "__main__":
    # First train the models if not already trained
    from pathlib import Path
    model_path = Path(__file__).parent / "data" / "stage2_model.joblib"

    if not model_path.exists():
        print("Models not found. Training first...\n")
        from train import train_all_models
        train_all_models(n_samples=30000)  # Smaller for demo speed

    run_demo()
