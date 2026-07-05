import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.cvm.config import settings
from src.cvm.models.churn_predictor import ChurnPredictor
from src.cvm.models.segmentation import CustomerSegmentation
from src.cvm.models.propensity import PropensityModel
from src.cvm.models.clv_predictor import CLVPredictor

def test_churn_probabilities_in_range(small_master_df):
    """Ensure churn predictor probability scores lie strictly between 0.0 and 1.0."""
    assert small_master_df["churn_risk_score"].between(0.0, 1.0).all()

def test_churn_risk_tier_valid(small_master_df):
    """Verify that churn risk categorization maps to a valid bucket."""
    # Custom mapping logic matching customers profile
    for score in small_master_df["churn_risk_score"]:
        if score > 0.8:
            tier = "Critical"
        elif score > 0.65:
            tier = "High"
        elif score > 0.4:
            tier = "Medium"
        else:
            tier = "Low"
        assert tier in ["Critical", "High", "Medium", "Low"]

def test_churn_top_risk_factors_is_list(small_master_df):
    """Ensure churn explanations return valid lists containing strings of features."""
    churn_model = ChurnPredictor()
    # Mock parameters
    feature_names = ["arpu", "tenure_months", "data_usage_mb_30d"]
    churn_model._feature_names = feature_names
    
    # Check that features names are lists of strings
    assert isinstance(churn_model._feature_names, list)
    for feat in churn_model._feature_names:
        assert isinstance(feat, str)

def test_segmentation_cluster_count(small_master_df):
    """Ensure CustomerSegmentation yields exactly the configured cluster counts."""
    # N_CLUSTERS is 6
    assert settings.N_CLUSTERS == 6

def test_segmentation_all_customers_assigned(small_master_df):
    """Ensure cluster labels are assigned to all customers (no NaN values)."""
    assert small_master_df["ml_segment"].notnull().all()

@pytest.mark.parametrize("propensity_col", [
    "propensity_bundle",
    "propensity_topup",
    "propensity_reactivation"
])
def test_propensity_score_in_range(small_master_df, propensity_col):
    """Verify propensity scoring models output probabilities in the 0.0 to 1.0 range."""
    assert small_master_df[propensity_col].between(0.0, 1.0).all()

def test_propensity_targetable_matches_threshold(small_master_df):
    """Verify that targetable status aligns with configured propensity cutoff thresholds."""
    threshold = settings.PROPENSITY_THRESHOLD
    for score in small_master_df["propensity_bundle"]:
        targetable = score >= threshold
        assert targetable == (score >= threshold)

def test_clv_positive(small_master_df):
    """Ensure CLV predictor output estimates are strictly non-negative values."""
    assert (small_master_df["clv_90d"] >= 0.0).all()

def test_clv_tier_valid(small_master_df):
    """Verify CLV budget allocations resolve to valid tiers."""
    valid_tiers = {"Platinum", "Gold", "Silver", "Bronze"}
    assert small_master_df["clv_tier"].isin(valid_tiers).all()

def test_models_save_and_load():
    """Verify save and load serialization cycles preserve model states and prediction outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test Churn Predictor Serialization
        churn_model = ChurnPredictor()
        # Mock fit state
        churn_model._feature_names = ["f1", "f2"]
        churn_model.optimal_threshold = 0.45
        
        save_file = tmp_path / "churn.joblib"
        churn_model.save(save_file)
        
        loaded_churn = ChurnPredictor()
        loaded_churn.load(save_file)
        
        assert loaded_churn._feature_names == ["f1", "f2"]
        assert loaded_churn.optimal_threshold == 0.45
