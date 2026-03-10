"""
Training Script — BNPL Decision Engine
────────────────────────────────────────
Trains all 3 models:
 
  2. Stage 1 Logistic Regression (pre-qualification)
  3. Stage 2 XGBoost (full credit decision)

To train on real data:
  Replace generate_synthetic_data() with your actual dataset loader.
  Recommended dataset: Home Credit Default Risk (Kaggle)
  Download: kaggle competitions download -c home-credit-default-risk

Usage:
  python train.py                        # Train on synthetic data
  python train.py --dataset home_credit  # Train on Home Credit dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.feature_engineering import ApplicationData, Stage1FeatureExtractor, Stage2FeatureExtractor
from models.stage1_logistic import Stage1Model
from models.stage2_xgboost import Stage2Model



def generate_synthetic_data(n_samples: int = 50000) -> pd.DataFrame:
    """
    Generate realistic synthetic credit data for training.
    Replace this with your real dataset for production use.
    
    Label distribution: ~8% default rate (realistic for BNPL)
    """
    print(f"Generating {n_samples:,} synthetic training samples...")
    rng = np.random.default_rng(42)

    n = n_samples
    data = {}

    # ── Applicant ──────────────────────────────────────────────────
    data["age"] = rng.integers(18, 72, n)
    data["annual_income"] = np.exp(rng.normal(10.8, 0.6, n)).clip(15000, 500000)
    data["employment_years"] = rng.exponential(4, n).clip(0, 35)
    data["employment_status"] = rng.choice(
        ["employed", "self_employed", "student", "unemployed"],
        n, p=[0.68, 0.15, 0.10, 0.07]
    )
    data["monthly_debt_obligations"] = data["annual_income"] / 12 * rng.uniform(0.05, 0.6, n)

    # ── Order ──────────────────────────────────────────────────────
    data["order_amount"] = rng.lognormal(5.5, 1.0, n).clip(20, 5000)
    data["merchant_category"] = rng.choice(
        ["grocery", "fashion", "electronics", "luxury", "travel"],
        n, p=[0.15, 0.30, 0.35, 0.10, 0.10]
    )
    data["installment_plan"] = rng.choice([3, 6, 12], n, p=[0.35, 0.45, 0.20])

    # ── Bureau ─────────────────────────────────────────────────────
    base_scores = (
        600
        + (data["annual_income"] - 50000) / 5000
        + data["employment_years"] * 3
        + rng.normal(0, 40, n)
    ).clip(300, 850)
    data["credit_score"] = base_scores

    data["credit_utilization"] = rng.beta(2, 5, n).clip(0.01, 0.99)
    data["num_delinquencies_2yr"] = rng.choice([0, 1, 2, 3, 4], n, p=[0.72, 0.14, 0.08, 0.04, 0.02])
    data["num_hard_inquiries_6mo"] = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.40, 0.30, 0.15, 0.08, 0.05, 0.02])
    data["num_open_accounts"] = rng.integers(1, 20, n)
    data["oldest_account_years"] = rng.exponential(6, n).clip(0.1, 30)
    data["total_credit_limit"] = (data["annual_income"] * rng.uniform(0.1, 1.5, n)).clip(500, 100000)
    data["total_revolving_balance"] = data["total_credit_limit"] * data["credit_utilization"]
    data["num_public_records"] = rng.choice([0, 1, 2, 3], n, p=[0.88, 0.08, 0.03, 0.01])

    # ── Behavioral ─────────────────────────────────────────────────
    data["device_age_days"] = rng.exponential(365, n).clip(0, 3650)
    data["session_duration_seconds"] = rng.lognormal(4.2, 0.8, n).clip(5, 600)
    data["email_domain_age_days"] = rng.exponential(730, n).clip(0, 7300)
    data["previous_bnpl_orders"] = rng.choice([0, 1, 2, 3, 4, 5, 10], n, p=[0.45, 0.20, 0.15, 0.08, 0.06, 0.04, 0.02])
    data["previous_bnpl_defaults"] = np.where(
        data["previous_bnpl_orders"] > 0,
        rng.binomial(data["previous_bnpl_orders"], 0.05),
        0
    )

    df = pd.DataFrame(data)

    # ── Generate Labels (default = 1) ──────────────────────────────
    # Logistic model for realistic default probability
    log_odds = (
        -6.0                                                  # base intercept
        + (df["num_delinquencies_2yr"] * 0.8)               # biggest risk factor
        + (df["credit_utilization"] * 2.5)                  # high utilization
        + (df["num_hard_inquiries_6mo"] * 0.3)              # many inquiries
        + (df["num_public_records"] * 1.2)                  # bankruptcies etc
        - ((df["credit_score"] - 600) / 100)                # good score reduces risk
        - (df["employment_years"] * 0.05)                   # stability reduces risk
        - (df["oldest_account_years"] * 0.04)               # history reduces risk
        + (df["monthly_debt_obligations"] / (df["annual_income"] / 12) * 1.5)  # DTI
        + (df["previous_bnpl_defaults"] * 1.5)              # prior defaults
        + rng.normal(0, 0.5, n)                             # noise
    )

    prob_default = 1 / (1 + np.exp(-log_odds))
    df["default"] = (rng.uniform(0, 1, n) < prob_default).astype(int)

    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Features: {len(df.columns) - 1}")
    return df


def build_feature_matrices(df: pd.DataFrame):
    """Convert raw dataframe to feature matrices for Stage 1 and Stage 2"""
    stage1_ext = Stage1FeatureExtractor()
    stage2_ext = Stage2FeatureExtractor()

    stage1_rows = []
    stage2_rows = []

    print("Extracting features...")
    for _, row in df.iterrows():
        app = ApplicationData(
            applicant_id=str(row.name),
            age=int(row["age"]),
            annual_income=float(row["annual_income"]),
            monthly_debt_obligations=float(row["monthly_debt_obligations"]),
            employment_years=float(row["employment_years"]),
            employment_status=str(row["employment_status"]),
            order_amount=float(row["order_amount"]),
            merchant_category=str(row["merchant_category"]),
            installment_plan=int(row["installment_plan"]),
            credit_score=float(row["credit_score"]),
            num_open_accounts=int(row["num_open_accounts"]),
            num_delinquencies_2yr=int(row["num_delinquencies_2yr"]),
            credit_utilization=float(row["credit_utilization"]),
            num_hard_inquiries_6mo=int(row["num_hard_inquiries_6mo"]),
            oldest_account_years=float(row["oldest_account_years"]),
            total_credit_limit=float(row["total_credit_limit"]),
            total_revolving_balance=float(row["total_revolving_balance"]),
            num_public_records=int(row["num_public_records"]),
            device_age_days=int(row["device_age_days"]),
            session_duration_seconds=int(row["session_duration_seconds"]),
            email_domain_age_days=int(row["email_domain_age_days"]),
            previous_bnpl_orders=int(row["previous_bnpl_orders"]),
            previous_bnpl_defaults=int(row["previous_bnpl_defaults"]),
        )
        stage1_rows.append(stage1_ext.extract(app))
        stage2_rows.append(stage2_ext.extract(app))

    stage1_df = pd.DataFrame(stage1_rows)
    stage2_df = pd.DataFrame(stage2_rows)
    labels = df["default"].values

    return stage1_df, stage2_df, labels, stage1_ext.get_feature_names(), stage2_ext.get_feature_names()


def train_all_models(n_samples: int = 50000):
    """Train all 3 models in the correct order"""

    # 1. Generate / load data
    df = generate_synthetic_data(n_samples)

    # 2. Extract features
    s1_df, s2_df, labels, s1_features, s2_features = build_feature_matrices(df)
    X_s1 = s1_df.values.astype(np.float32)
    X_s2 = s2_df.values.astype(np.float32)

    # 3. Train/val split (80/20, stratified by default label)
    idx = np.arange(len(labels))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)

    X_s1_train, X_s1_val = X_s1[tr_idx], X_s1[val_idx]
    X_s2_train, X_s2_val = X_s2[tr_idx], X_s2[val_idx]
    y_train, y_val = labels[tr_idx], labels[val_idx]

    # 4. Train Fraud Detector on non-default samples (it learns "normal")
    print("\n" + "="*60)
    

    # 5. Train Stage 1 — Logistic Regression
    print()
    stage1 = Stage1Model()
    s1_auc = stage1.train(X_s1_train, y_train, X_s1_val, y_val, s1_features)

    # 6. Train Stage 2 — XGBoost with monotone constraints
    print()
    stage2 = Stage2Model()

    # Define monotone constraints for regulatory compliance
    # +1 = higher value → more risk (monotone increasing)
    # -1 = higher value → less risk (monotone decreasing)
    monotone_constraints = {
        "credit_score_normalized":    -1,  # higher score = less risk
        "num_delinquencies_2yr":      +1,  # more delinquencies = more risk
        "credit_utilization":         +1,  # higher utilization = more risk
        "num_hard_inquiries_6mo":     +1,  # more inquiries = more risk
        "num_public_records":         +1,  # more records = more risk
        "debt_to_income_ratio":       +1,  # higher DTI = more risk
        "payment_history_score":      -1,  # better payment history = less risk
        "oldest_account_years":       -1,  # longer history = less risk
        "employment_years":           -1,  # more experience = less risk
    }

    s2_auc = stage2.train(
        X_s2_train, y_train, X_s2_val, y_val, s2_features, monotone_constraints
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Stage 1 AUC: {s1_auc:.4f}")
    print(f"Stage 2 AUC: {s2_auc:.4f}")
    print("\nAll models saved to bnpl_engine/data/")
    print("Run demo.py to test the engine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BNPL Decision Engine")
    parser.add_argument("--samples", type=int, default=50000, help="Training samples to generate")
    args = parser.parse_args()

    train_all_models(n_samples=args.samples)
