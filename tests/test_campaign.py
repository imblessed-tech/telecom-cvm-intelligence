import pytest
import pandas as pd

from src.cvm.config import settings
from src.cvm.campaign.opportunity_base import OpportunityBaseGenerator

def test_opportunity_base_not_empty(small_master_df):
    """Ensure generated campaign bases return records when suitable profiles exist."""
    # Force some high values to guarantee eligible customers
    small_master_df.loc[0, "churn_risk_score"] = 0.95
    small_master_df.loc[0, "clv_tier"] = "Platinum"
    small_master_df.loc[0, "tenure_months"] = 48
    small_master_df.loc[0, "customer_id"] = "test_eligible_id"
    
    generator = OpportunityBaseGenerator(small_master_df)
    # Clear suppressed IDs for deterministic checks
    generator.suppressed_ids = set()
    
    opp_df = generator.generate("churn_retention", max_size=5)
    assert len(opp_df) > 0
    assert (opp_df["customer_id"] == "test_eligible_id").any()

def test_opportunity_base_max_size_respected(small_master_df):
    """Ensure opportunity base sizes respect the max_size constraint."""
    generator = OpportunityBaseGenerator(small_master_df)
    generator.suppressed_ids = set()
    
    max_size = 3
    opp_df = generator.generate("churn_retention", max_size=max_size)
    assert len(opp_df) <= max_size

def test_churn_retention_filters_low_risk(small_master_df):
    """Ensure customers below the churn threshold are excluded from retention campaigns."""
    generator = OpportunityBaseGenerator(small_master_df)
    generator.suppressed_ids = set()
    
    opp_df = generator.generate("churn_retention", max_size=100)
    if len(opp_df) > 0:
        assert (opp_df["churn_risk_score"] >= settings.CHURN_RISK_THRESHOLD).all()

def test_bundle_upsell_excludes_high_churn(small_master_df):
    """Ensure customers with high churn risk are excluded from bundle upsells (retention first)."""
    # Force some churn risks to test exclusion
    small_master_df["propensity_bundle"] = 0.90
    small_master_df.loc[0, "churn_risk_score"] = 0.75
    
    generator = OpportunityBaseGenerator(small_master_df)
    generator.suppressed_ids = set()
    
    opp_df = generator.generate("bundle_upsell", max_size=100)
    if len(opp_df) > 0:
        assert (opp_df["churn_risk_score"] < 0.5).all()

def test_priority_score_descending(small_master_df):
    """Ensure that campaign targets are sorted by priority_score descending."""
    generator = OpportunityBaseGenerator(small_master_df)
    generator.suppressed_ids = set()
    
    opp_df = generator.generate("churn_retention", max_size=50)
    if len(opp_df) > 1:
        priority_scores = opp_df["priority_score"].tolist()
        assert all(priority_scores[i] >= priority_scores[i+1] for i in range(len(priority_scores)-1))

def test_recommended_channel_valid(small_master_df):
    """Verify that recommendations resolve to one of the valid channels."""
    generator = OpportunityBaseGenerator(small_master_df)
    generator.suppressed_ids = set()
    
    valid_channels = {"SMS", "USSD", "PUSH_NOTIFICATION", "OUTBOUND_CALL"}
    
    for ctype in ["churn_retention", "bundle_upsell", "voice_topup", "reactivation", "loyalty_reward"]:
        opp_df = generator.generate(ctype, max_size=50)
        if len(opp_df) > 0:
            assert opp_df["recommended_channel"].isin(valid_channels).all()
