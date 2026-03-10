"""
Training Script — BNPL Decision Engine
────────────────────────────────────────
Trains both models:
  1. Stage 1 Logistic Regression (pre-qualification)
  2. Stage 2 XGBoost (full credit decision)

To train on real data:
  Replace generate_synthetic_data() with your actual dataset loader.
  Recommended dataset: Home Credit Default Risk (Kaggle)
  Download: kaggle competitions download -c home-credit-default-risk

Usage:
  python train.py                   # Train on synthetic data
  python train.py --samples 100000  # Train with more samples
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


def load_kaggle_data(n_samples: int = 50000) -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "home-credit-default-risk", "application_train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Kaggle dataset not found at {csv_path}. Please download it first.")

    print(f"Loading {n_samples:,} samples from Kaggle dataset ({csv_path})...")
    
    # We only read the rows we need to save memory 
    # (or sample from the whole file if n_samples < 307511)
    df_raw = pd.read_csv(csv_path, nrows=n_samples)
    
    rng = np.random.default_rng(42)
    n = len(df_raw)
    data = {}

    # --- Applicant Identity & Financials (Real Kaggle Data) ---
    data["applicant_id"] = df_raw["SK_ID_CURR"]
    data["age"] = (df_raw["DAYS_BIRTH"] / -365).astype(int)
    data["annual_income"] = df_raw["AMT_INCOME_TOTAL"]
    
    # Handle the '365243' DAYS_EMPLOYED outlier which means unemployed/pensioner
    days_employed_cleaned = df_raw["DAYS_EMPLOYED"].replace(365243, 0)
    data["employment_years"] = (days_employed_cleaned / -365).clip(lower=0)
    
    # Map employment status
    income_map = {
        "Working": "employed",
        "Commercial associate": "self_employed",
        "Pensioner": "unemployed",
        "State servant": "employed",
        "Student": "student",
        "Unemployed": "unemployed",
        "Businessman": "self_employed",
        "Maternity leave": "unemployed"
    }
    data["employment_status"] = df_raw["NAME_INCOME_TYPE"].map(income_map).fillna("employed")
    
    # Debt/Loan mapping
    data["monthly_debt_obligations"] = df_raw["AMT_ANNUITY"].fillna(data["annual_income"] * 0.05)
    data["order_amount"] = df_raw["AMT_CREDIT"]

    # --- Bureau Data (Partial Real / Partial Synthetic) ---
    # We create a synthesized credit score based on Kaggle's normalized external sources
    ext_sources = df_raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].fillna(0.5).mean(axis=1)
    # Ext sources range 0 to 1, where higher is lower risk (usually). Map to 300-850 FICO score.
    # Note: Kaggle ext_sources are highly correlated with the target
    data["credit_score"] = (300 + (ext_sources * 550)).clip(300, 850)
    
    data["num_hard_inquiries_6mo"] = (df_raw["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0) + df_raw["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0)).astype(int)
    
    # Synthesize remaining Bureau fields based on actual Kaggle anchors
    data["credit_utilization"] = rng.beta(2, 5, n).clip(0.01, 0.99)
    data["num_delinquencies_2yr"] = rng.choice([0, 1, 2, 3, 4], n, p=[0.72, 0.14, 0.08, 0.04, 0.02])
    data["num_open_accounts"] = rng.integers(1, 20, n)
    data["oldest_account_years"] = data["age"] / 2.0  # Safe heuristic 
    data["total_credit_limit"] = (data["annual_income"] * rng.uniform(0.1, 1.5, n)).clip(500, 100000)
    data["total_revolving_balance"] = data["total_credit_limit"] * data["credit_utilization"]
    data["num_public_records"] = rng.choice([0, 1, 2, 3], n, p=[0.88, 0.08, 0.03, 0.01])

    # --- Order Context & Specific BNPL Anti-Fraud Signals (Synthetic) ---
    # Because Kaggle is for massive standard loans (mortgages, cars), not small cart checkouts
    data["merchant_category"] = rng.choice(["grocery", "fashion", "electronics", "luxury", "travel"], n, p=[0.15, 0.30, 0.35, 0.10, 0.10])
    data["installment_plan"] = rng.choice([3, 6, 12], n, p=[0.35, 0.45, 0.20])
    
    data["device_age_days"] = rng.exponential(365, n).clip(0, 3650)
    data["session_duration_seconds"] = rng.lognormal(4.2, 0.8, n).clip(5, 600)
    data["email_domain_age_days"] = rng.exponential(730, n).clip(0, 7300)
    data["previous_bnpl_orders"] = rng.choice([0, 1, 2, 3], n, p=[0.50, 0.30, 0.15, 0.05])
    data["previous_bnpl_defaults"] = np.where(data["previous_bnpl_orders"] > 0, rng.binomial(data["previous_bnpl_orders"], 0.05), 0)

    df = pd.DataFrame(data)
    
    # Set ID as index
    df = df.set_index("applicant_id")

    # The actual Ground Truth label from the Kaggle dataset
    df["default"] = df_raw["TARGET"].values

    print(f"Loaded Real Data - Default rate: {df['default'].mean():.2%}")
    print(f"Features ready for extraction: {len(df.columns) - 1}")
    return df


def build_feature_matrices(df: pd.DataFrame):
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
    # 1. Generate data
    df = load_kaggle_data(n_samples)

    # 2. Extract features
    s1_df, s2_df, labels, s1_features, s2_features = build_feature_matrices(df)
    X_s1 = s1_df.values.astype(np.float32)
    X_s2 = s2_df.values.astype(np.float32)

    # 3. Train/val split
    idx = np.arange(len(labels))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)

    X_s1_train, X_s1_val = X_s1[tr_idx], X_s1[val_idx]
    X_s2_train, X_s2_val = X_s2[tr_idx], X_s2[val_idx]
    y_train, y_val = labels[tr_idx], labels[val_idx]

    # 4. Train Stage 1 — Logistic Regression
    print("=" * 60)
    stage1 = Stage1Model()
    s1_auc = stage1.train(X_s1_train, y_train, X_s1_val, y_val, s1_features)

    # 5. Train Stage 2 — XGBoost
    print()
    stage2 = Stage2Model()

    monotone_constraints = {
        "credit_score_normalized":  -1,
        "num_delinquencies_2yr":    +1,
        "credit_utilization":       +1,
        "num_hard_inquiries_6mo":   +1,
        "num_public_records":       +1,
        "debt_to_income_ratio":     +1,
        "payment_history_score":    -1,
        "oldest_account_years":     -1,
        "employment_years":         -1,
    }

    s2_auc = stage2.train(
        X_s2_train, y_train, X_s2_val, y_val, s2_features, monotone_constraints
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Stage 1 AUC : {s1_auc:.4f}")
    print(f"Stage 2 AUC : {s2_auc:.4f}")
    print("\nModels saved to data/")
    print("Run demo.py to test the engine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BNPL Decision Engine")
    parser.add_argument("--samples", type=int, default=50000, help="Number of training samples")
    args = parser.parse_args()
    train_all_models(n_samples=args.samples)