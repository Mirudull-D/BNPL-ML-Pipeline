"""
Demo Script — Test the BNPL Decision Engine on REAL Kaggle Data
Run: python demo_kaggle.py

Pulls 5 random applicants directly from the home-credit-default-risk CSV.
"""

import sys, os, pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.feature_engineering import ApplicationData
from pipeline.decision_engine import DecisionEngine
from train import load_kaggle_data
from demo import print_result

def run_kaggle_demo():
    print("Loading 5 applicants directly from Kaggle CSV...")
    
    # Load just a few rows to get real application data
    df_real = load_kaggle_data(n_samples=5000)
    
    # Pick 5 interesting cases (mix of defaults and non-defaults if possible)
    # df_real has the index as 'applicant_id'
    
    defaults = df_real[df_real['default'] == 1].head(2)
    repaids = df_real[df_real['default'] == 0].head(3)
    
    test_rows = pd.concat([repaids, defaults]).sample(frac=1, random_state=42) # Shuffle them
    
    engine = DecisionEngine()
    
    print("\n" + "="*60)
    print("  BNPL Decision Engine — REAL KAGGLE DATA DEMO")
    print("="*60)

    for i, (applicant_id, row) in enumerate(test_rows.iterrows()):
        
        # Build the exact ApplicationData object from the loaded Kaggle row
        app = ApplicationData(
            applicant_id=f"KAGGLE_{applicant_id}",
            age=int(row["age"]),
            annual_income=float(row["annual_income"]),
            monthly_debt_obligations=float(row["monthly_debt_obligations"]),
            employment_years=float(row["employment_years"]),
            employment_status=str(row["employment_status"]),
            order_amount=float(row["order_amount"]),
            merchant_category=str(row["merchant_category"]),
            installment_plan=int(row["installment_plan"]),
            
            # The real bureau stats from the Kaggle row used in Stage 2
            credit_score=float(row.get("credit_score", 600)),
            num_open_accounts=int(row.get("num_open_accounts", 5)),
            num_delinquencies_2yr=int(row.get("num_delinquencies_2yr", 0)),
            credit_utilization=float(row.get("credit_utilization", 0.3)),
            num_hard_inquiries_6mo=int(row.get("num_hard_inquiries_6mo", 1)),
            oldest_account_years=float(row.get("oldest_account_years", 5.0)),
            total_credit_limit=float(row.get("total_credit_limit", 5000)),
            total_revolving_balance=float(row.get("total_revolving_balance", 1000)),
            num_public_records=int(row.get("num_public_records", 0)),
            
            # Behavioral
            device_age_days=int(row.get("device_age_days", 365)),
            session_duration_seconds=int(row.get("session_duration_seconds", 120)),
            email_domain_age_days=int(row.get("email_domain_age_days", 500)),
            previous_bnpl_orders=int(row.get("previous_bnpl_orders", 0)),
            previous_bnpl_defaults=int(row.get("previous_bnpl_defaults", 0))
        )

        truth = "DEFAULTED" if row['default'] == 1 else "REPAID"
        print(f"\n{'─'*60}")
        print(f"  Applicant {i+1}: ID {applicant_id}  |  Actual Kaggle Truth: {truth}")
        print(f"  Income: ₹{app.annual_income:,.0f} | Age: {app.age} | Employed: {app.employment_years:.1f} yrs")
        
        # We set simulate_bureau=False because we ALREADY loaded the real bureau
        # stats into the ApplicationData object directly from the Kaggle CSV just now.
        result = engine.evaluate(
            app,
            simulate_bureau=False 
        )
        print_result(result)

    print(f"\n{'='*60}")
    print("  Kaggle Demo complete.")
    print("="*60)

if __name__ == "__main__":
    run_kaggle_demo()
