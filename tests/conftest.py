import sys
from pathlib import Path

# Add project root and src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.cvm.api.dependencies import CVMModelRegistry

@pytest.fixture
def sample_ibm_df():
    """Generates a mock IBM Telecom Churn DataFrame for unit testing."""
    np.random.seed(42)
    n_rows = 50
    customer_ids = [f"cust_{i:04d}" for i in range(n_rows)]
    
    data = {
        "customerID": customer_ids,
        "gender": np.random.choice(["Male", "Female"], n_rows),
        "SeniorCitizen": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        "Partner": np.random.choice(["Yes", "No"], n_rows),
        "Dependents": np.random.choice(["Yes", "No"], n_rows),
        "tenure": np.random.randint(1, 72, n_rows),
        "PhoneService": np.random.choice(["Yes", "No"], n_rows),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_rows),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_rows),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_rows),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_rows),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_rows),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_rows),
        "MonthlyCharges": np.random.uniform(20.0, 120.0, n_rows),
        "TotalCharges": np.random.uniform(100.0, 5000.0, n_rows).astype(str),  # IBM schema TotalCharges is string
        "Churn": np.random.choice(["Yes", "No"], n_rows, p=[0.26, 0.74])
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_events_df():
    """Generates a mock transactional events DataFrame for unit testing."""
    np.random.seed(42)
    n_customers = 50
    n_events = 500
    customer_ids = [f"cust_{i:04d}" for i in range(n_customers)]
    
    dates = pd.date_range(end="2026-07-01", periods=90).strftime("%Y-%m-%d").tolist()
    
    event_types = ["RECHARGE", "DATA_SESSION", "OUTGOING_CALL", "COMPLAINT", "APP_LOGIN"]
    
    data = {
        "event_id": [f"evt_{i:06d}" for i in range(n_events)],
        "customer_id": np.random.choice(customer_ids, n_events),
        "event_type": np.random.choice(event_types, n_events, p=[0.3, 0.4, 0.15, 0.05, 0.1]),
        "event_date": np.random.choice(dates, n_events),
        "recharge_amount": np.random.choice([200.0, 500.0, 1000.0, 2000.0, 5000.0], n_events),
        "data_mb_used": np.random.uniform(5.0, 500.0, n_events)
    }
    return pd.DataFrame(data)

@pytest.fixture
def small_master_df():
    """Generates an enriched customer feature DataFrame mimicking output of preprocessors and engineers."""
    np.random.seed(42)
    n_rows = 50
    customer_ids = [f"cust_{i:04d}" for i in range(n_rows)]
    
    stages = ["New Active", "Active", "High Value Loyal", "Lapsing", "Churned"]
    rfm_segs = ["Champions", "Loyal Customers", "Potential Loyalist", "At Risk", "Cant Lose Them", "Lost"]
    clv_tiers = ["Platinum", "Gold", "Silver", "Bronze"]
    
    data = {
        "customer_id": customer_ids,
        "gender": np.random.choice(["Male", "Female"], n_rows),
        "tenure_months": np.random.randint(1, 72, n_rows),
        "monthly_charges": np.random.uniform(100.0, 15000.0, n_rows),
        "total_charges": np.random.uniform(1000.0, 50000.0, n_rows),
        "churn_label": np.random.choice([0, 1], n_rows, p=[0.75, 0.25]),
        "recharge_count_90d": np.random.randint(0, 30, n_rows),
        "total_recharge_90d": np.random.uniform(500.0, 25000.0, n_rows),
        "avg_recharge_amount": np.random.uniform(200.0, 3000.0, n_rows),
        "days_since_last_recharge": np.random.randint(0, 90, n_rows),
        "data_usage_mb_90d": np.random.uniform(0.0, 50000.0, n_rows),
        "data_usage_mb_30d": np.random.uniform(0.0, 20000.0, n_rows),
        "avg_session_mb": np.random.uniform(0.0, 500.0, n_rows),
        "active_days_90d": np.random.randint(0, 90, n_rows),
        "app_usage_rate": np.random.uniform(0.0, 1.0, n_rows),
        "recharge_trend": np.random.uniform(-10.0, 10.0, n_rows),
        "r_score": np.random.randint(1, 6, n_rows),
        "f_score": np.random.randint(1, 6, n_rows),
        "m_score": np.random.randint(1, 6, n_rows),
        "rfm_score": np.random.randint(3, 16, n_rows),
        "rfm_segment": np.random.choice(rfm_segs, n_rows),
        "lifecycle_stage": np.random.choice(stages, n_rows),
        "preferred_channel_ussd": np.random.choice([0, 1], n_rows),
        "preferred_channel_app": np.random.choice([0, 1], n_rows),
        "preferred_channel_agent": np.random.choice([0, 1], n_rows),
        "preferred_channel_web": np.random.choice([0, 1], n_rows),
        "arpu": np.random.uniform(200.0, 8000.0, n_rows),
        "engagement_score": np.random.uniform(0.0, 1.0, n_rows),
        "declining_recharge": np.random.choice([0, 1], n_rows),
        "long_silence": np.random.choice([0, 1], n_rows),
        "low_activity": np.random.choice([0, 1], n_rows),
        "churn_risk_score": np.random.uniform(0.05, 0.95, n_rows),
        "ml_segment": np.random.choice(["High Spend Data Users", "Low Value Voice", "Price Sensitive", "Dormant", "At Risk", "Rising Stars"], n_rows),
        "propensity_bundle": np.random.uniform(0.05, 0.95, n_rows),
        "propensity_topup": np.random.uniform(0.05, 0.95, n_rows),
        "propensity_reactivation": np.random.uniform(0.05, 0.95, n_rows),
        "clv_90d": np.random.uniform(100.0, 30000.0, n_rows),
        "clv_tier": np.random.choice(clv_tiers, n_rows)
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_registry():
    """Generates a mock model registry preloaded with Mock versions of all estimators."""
    registry = CVMModelRegistry()
    registry.preprocessor = MagicMock()
    
    # Churn predictor mock
    churn = MagicMock()
    churn._feature_names = ["arpu", "tenure_months", "data_usage_mb_30d"]
    churn.optimal_threshold = 0.5
    
    # Shap Explainer Mock
    shap = MagicMock()
    shap.shap_values = MagicMock(return_value=np.array([[0.1, -0.05, 0.15]]))
    churn.shap_explainer = shap
    
    class Pred:
        def __init__(self):
            self.churn_probability = 0.75
            self.churn_predicted = True
            self.segment_label = "High Spend Data Users"
            self.propensity_score = 0.85
            self.predicted_clv_90d = 5000.0
            self.clv_tier = "Platinum"
            self.retention_budget_tier = "Premium"
            
    churn.predict = MagicMock(return_value=[Pred(), Pred()])
    churn.predict_proba = MagicMock(return_value=np.array([[0.25, 0.75], [0.8, 0.2]]))
    registry.churn_model = churn
    
    # Segmentation mock
    seg = MagicMock()
    seg.is_fitted = True
    seg.predict = MagicMock(return_value=[Pred(), Pred()])
    seg.cluster_profiles = pd.DataFrame(index=[0, 1, 2, 3, 4, 5])
    seg._campaign_mapping = {
        0: {"label": "High Spend Data Users", "description": "Desc 0", "recommended_campaign": "bundle_upsell"},
        1: {"label": "Low Value Voice", "description": "Desc 1", "recommended_campaign": "voice_topup"},
        2: {"label": "Price Sensitive", "description": "Desc 2", "recommended_campaign": "voice_topup"},
        3: {"label": "Dormant", "description": "Desc 3", "recommended_campaign": "reactivation"},
        4: {"label": "At Risk", "description": "Desc 4", "recommended_campaign": "churn_retention"},
        5: {"label": "Rising Stars", "description": "Desc 5", "recommended_campaign": "bundle_upsell"}
    }
    registry.segmentation_model = seg
    
    # Propensity models mocks
    registry.propensity_bundle = MagicMock()
    registry.propensity_bundle.predict = MagicMock(return_value=[Pred(), Pred()])
    
    registry.propensity_topup = MagicMock()
    registry.propensity_topup.predict = MagicMock(return_value=[Pred(), Pred()])
    
    registry.propensity_reactivation = MagicMock()
    registry.propensity_reactivation.predict = MagicMock(return_value=[Pred(), Pred()])
    
    # CLV mock
    clv = MagicMock()
    clv.predict = MagicMock(return_value=[Pred(), Pred()])
    registry.clv_model = clv
    
    # Master DF mock
    registry.master_df = pd.DataFrame({
        "customer_id": ["cust_0001", "cust_0002"],
        "tenure_months": [12, 24],
        "monthly_charges": [200.0, 1500.0],
        "total_charges": [2400.0, 36000.0],
        "churn_risk_score": [0.75, 0.20],
        "clv_90d": [5000.0, 1000.0],
        "clv_tier": ["Platinum", "Bronze"],
        "rfm_segment": ["Champions", "Lost"],
        "ml_segment": ["High Spend Data Users", "Dormant"],
        "propensity_bundle": [0.85, 0.10],
        "propensity_topup": [0.60, 0.20],
        "propensity_reactivation": [0.10, 0.70],
        "lifecycle_stage": ["Active", "Lapsing"],
        "r_score": [5, 1],
        "f_score": [5, 1],
        "m_score": [5, 1],
        "data_usage_mb_30d": [4500.0, 120.0],
        "app_usage_rate": [0.4, 0.05],
        "preferred_channel_ussd": [0, 1],
        "preferred_channel_app": [1, 0]
    })
    
    registry.is_loaded = True
    return registry
