"""
Feature Engineering Pipeline
Transforms raw application data into model-ready features for Stage 1 & Stage 2
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ApplicationData:
    """Raw input from credit application form"""
    # Applicant identity
    applicant_id: str
    age: int

    # Financial (stated on application)
    annual_income: float
    monthly_debt_obligations: float
    employment_years: float
    employment_status: str          # "employed", "self_employed", "unemployed", "student"

    # Order details
    order_amount: float
    merchant_category: str          # "electronics", "fashion", "grocery", "luxury", "travel"
    installment_plan: int           # 3, 6, 12 months

    # Bureau data (Stage 2 only - fetched after Stage 1 approval)
    credit_score: float = None
    num_open_accounts: int = None
    num_delinquencies_2yr: int = None
    credit_utilization: float = None
    num_hard_inquiries_6mo: int = None
    oldest_account_years: float = None
    total_credit_limit: float = None
    total_revolving_balance: float = None
    num_public_records: int = None  # bankruptcies, liens

    # Behavioral / Alternative data
    device_age_days: int = None
    session_duration_seconds: int = None
    email_domain_age_days: int = None
    previous_bnpl_orders: int = 0
    previous_bnpl_defaults: int = 0


class Stage1FeatureExtractor:
    """
    Stage 1: Application-only features (NO bureau pull).
    Fast, cheap, used for instant pre-qualification.
    Target latency: < 5ms
    """

    MERCHANT_RISK = {
        "grocery": 0.1,
        "fashion": 0.3,
        "electronics": 0.5,
        "travel": 0.6,
        "luxury": 0.8,
    }

    EMPLOYMENT_RISK = {
        "employed": 0.1,
        "self_employed": 0.4,
        "student": 0.6,
        "unemployed": 0.9,
    }

    def extract(self, app: ApplicationData) -> Dict[str, float]:
        """Extract Stage 1 features — no external API calls needed"""

        dti = (app.monthly_debt_obligations / (app.annual_income / 12)
               if app.annual_income > 0 else 1.0)

        order_to_income_ratio = (app.order_amount / (app.annual_income / 12)
                                  if app.annual_income > 0 else 1.0)

        installment_amount = app.order_amount / app.installment_plan
        installment_to_income = (installment_amount / (app.annual_income / 12)
                                  if app.annual_income > 0 else 1.0)

        bnpl_default_rate = (app.previous_bnpl_defaults / app.previous_bnpl_orders
                              if app.previous_bnpl_orders > 0 else 0.0)

        features = {
            # Income & Debt
            "annual_income_log": np.log1p(app.annual_income),
            "monthly_income": app.annual_income / 12,
            "debt_to_income_ratio": min(dti, 5.0),
            "monthly_debt_obligations": app.monthly_debt_obligations,

            # Order risk
            "order_amount_log": np.log1p(app.order_amount),
            "order_to_monthly_income_ratio": min(order_to_income_ratio, 10.0),
            "installment_to_income_ratio": min(installment_to_income, 5.0),
            "installment_plan_months": app.installment_plan,
            "merchant_risk_score": self.MERCHANT_RISK.get(app.merchant_category, 0.5),

            # Employment
            "employment_years": min(app.employment_years, 30),
            "employment_risk_score": self.EMPLOYMENT_RISK.get(app.employment_status, 0.5),
            "is_employed": 1.0 if app.employment_status == "employed" else 0.0,

            # Applicant
            "age": app.age,
            "age_risk": 1.0 if app.age < 22 or app.age > 70 else 0.0,

            # BNPL history
            "previous_bnpl_orders": min(app.previous_bnpl_orders, 50),
            "previous_bnpl_defaults": app.previous_bnpl_defaults,
            "bnpl_default_rate": bnpl_default_rate,
            "has_bnpl_history": 1.0 if app.previous_bnpl_orders > 0 else 0.0,

            # Behavioral / device signals
            "device_age_days": min(app.device_age_days or 0, 3650),
            "email_domain_age_days": min(app.email_domain_age_days or 0, 7300),
            "session_duration_seconds": min(app.session_duration_seconds or 0, 600),
            "device_is_new": 1.0 if (app.device_age_days or 0) < 7 else 0.0,
            "email_is_new": 1.0 if (app.email_domain_age_days or 0) < 30 else 0.0,
        }

        return features

    def get_feature_names(self):
        return list(self.extract(ApplicationData(
            applicant_id="x", age=30, annual_income=60000,
            monthly_debt_obligations=500, employment_years=3,
            employment_status="employed", order_amount=300,
            merchant_category="electronics", installment_plan=6,
        )).keys())


class Stage2FeatureExtractor:
    """
    Stage 2: Full feature set including bureau data.
    Only runs after Stage 1 pre-qualification.
    Target latency: < 20ms (bureau data already fetched)
    """

    def __init__(self):
        self.stage1 = Stage1FeatureExtractor()

    def extract(self, app: ApplicationData) -> Dict[str, float]:
        """Extract full feature set = Stage 1 features + bureau features"""

        # Start with all Stage 1 features
        features = self.stage1.extract(app)

        # --- Bureau Features ---
        credit_score = app.credit_score or 300
        utilization = app.credit_utilization or 1.0
        delinquencies = app.num_delinquencies_2yr or 0
        inquiries = app.num_hard_inquiries_6mo or 0
        open_accounts = app.num_open_accounts or 0
        oldest_acct = app.oldest_account_years or 0
        total_limit = app.total_credit_limit or 1
        revolving_bal = app.total_revolving_balance or 0
        public_records = app.num_public_records or 0

        # Derived bureau features
        available_credit = max(0, total_limit - revolving_bal)
        payment_history_score = max(0, 1.0 - (delinquencies * 0.2))

        bureau_features = {
            # Raw bureau
            "credit_score": credit_score,
            "credit_score_normalized": (credit_score - 300) / 550,  # normalize 300-850
            "credit_utilization": min(utilization, 1.0),
            "num_delinquencies_2yr": delinquencies,
            "num_hard_inquiries_6mo": inquiries,
            "num_open_accounts": open_accounts,
            "oldest_account_years": oldest_acct,
            "total_credit_limit_log": np.log1p(total_limit),
            "total_revolving_balance_log": np.log1p(revolving_bal),
            "num_public_records": public_records,

            # Derived bureau
            "available_credit_log": np.log1p(available_credit),
            "payment_history_score": payment_history_score,
            "has_delinquency": 1.0 if delinquencies > 0 else 0.0,
            "has_public_record": 1.0 if public_records > 0 else 0.0,
            "high_utilization": 1.0 if utilization > 0.7 else 0.0,
            "many_inquiries": 1.0 if inquiries >= 3 else 0.0,
            "thin_file": 1.0 if open_accounts <= 2 else 0.0,

            # Combined signals
            "credit_score_x_utilization": credit_score * (1 - min(utilization, 1.0)),
            "income_to_debt_ratio": (
                app.annual_income / max(app.monthly_debt_obligations * 12, 1)
            ),
            "credit_limit_to_income": total_limit / max(app.annual_income, 1),
        }

        features.update(bureau_features)
        return features

    def get_feature_names(self):
        app = ApplicationData(
            applicant_id="x", age=30, annual_income=60000,
            monthly_debt_obligations=500, employment_years=3,
            employment_status="employed", order_amount=300,
            merchant_category="electronics", installment_plan=6,
            credit_score=680, num_open_accounts=4,
            num_delinquencies_2yr=0, credit_utilization=0.3,
            num_hard_inquiries_6mo=1, oldest_account_years=5,
            total_credit_limit=15000, total_revolving_balance=4500,
            num_public_records=0,
        )
        return list(self.extract(app).keys())
