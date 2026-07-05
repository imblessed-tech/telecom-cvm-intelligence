import pytest
import numpy as np
import pandas as pd

from src.cvm.features.rfm import RFMAnalyser
from src.cvm.features.behavioural import BehaviouralFeatureEngineer

def test_rfm_scores_in_range(small_master_df):
    """Ensure that RFM scores fall strictly within the 1 to 5 index range."""
    assert small_master_df["r_score"].between(1, 5).all()
    assert small_master_df["f_score"].between(1, 5).all()
    assert small_master_df["m_score"].between(1, 5).all()

def test_rfm_segment_assigned_to_all_customers(small_master_df):
    """Ensure every customer has been assigned an RFM segment with no missing entries."""
    assert small_master_df["rfm_segment"].notnull().all()
    assert (small_master_df["rfm_segment"] != "").all()

def test_churn_signals_are_binary(small_master_df):
    """Ensure behavioral churn indicators (declining recharges, silence, low activity) are binary (0 or 1)."""
    for col in ["declining_recharge", "long_silence", "low_activity"]:
        assert small_master_df[col].isin([0, 1]).all()

def test_engagement_score_in_range(small_master_df):
    """Verify that overall engagement scores are bounded between 0.0 and 1.0 inclusive."""
    assert small_master_df["engagement_score"].between(0.0, 1.0).all()

def test_arpu_positive(small_master_df):
    """Ensure average revenue per user (ARPU) is strictly non-negative."""
    assert (small_master_df["arpu"] >= 0.0).all()

def test_lifecycle_stage_valid_values(small_master_df):
    """Verify that all customer lifecycle stages map to defined stages."""
    valid_stages = {"New Active", "Active", "High Value Loyal", "Lapsing", "Churned"}
    assert small_master_df["lifecycle_stage"].isin(valid_stages).all()

def test_recharge_trend_computable(sample_events_df, small_master_df):
    """Verify that the behavioral feature engineer can compute recharge trend values."""
    engineer = BehaviouralFeatureEngineer()
    # Apply calculation on small master
    result_df = engineer.engineer(small_master_df)
    assert "recharge_trend" in result_df.columns
    # Ensure there are non-null elements
    assert result_df["recharge_trend"].notnull().any()

def test_rfm_champions_have_high_scores(small_master_df):
    """Verify that customers categorized as 'Champions' possess high Recency and Frequency scores (R >= 4, F >= 4)."""
    champions = small_master_df[small_master_df["rfm_segment"] == "Champions"]
    if not champions.empty:
        assert (champions["r_score"] >= 4).all()
        assert (champions["f_score"] >= 4).all()
