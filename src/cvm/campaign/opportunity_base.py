from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from src.cvm.config import settings

logger = logging.getLogger(__name__)

@dataclass
class CampaignOpportunity:
    customer_id: str
    campaign_name: str
    segment_label: str
    rfm_segment: str
    churn_risk_score: float
    propensity_score: float
    clv_tier: str
    recommended_offer: str
    recommended_channel: str
    priority_score: float
    estimated_response_rate: float
    retention_budget: str

class OpportunityBaseGenerator:
    def __init__(self, master_df: pd.DataFrame):
        self.df = master_df.copy()
        
        # Simulate suppression by adding a random 10% of customer IDs to self.suppressed_ids
        np.random.seed(settings.RANDOM_STATE)
        all_ids = self.df["customer_id"].tolist()
        suppressed_count = int(len(all_ids) * 0.1)
        if suppressed_count > 0:
            self.suppressed_ids = set(np.random.choice(all_ids, size=suppressed_count, replace=False))
        else:
            self.suppressed_ids = set()
        logger.info(f"Initialized OpportunityBaseGenerator with {len(self.df)} customers. Simulated suppression list size: {len(self.suppressed_ids)}")

    def _assign_channel(self, row: pd.Series) -> str:
        """Assign recommended channel based on user usage and profile."""
        app_rate = row.get("app_usage_rate", 0.0)
        if app_rate > 0.3 or row.get("preferred_channel_app", 0) == 1:
            return "PUSH_NOTIFICATION"
            
        pref_channel = row.get("preferred_channel", "")
        if isinstance(pref_channel, str):
            pref_channel = pref_channel.upper()
            
        if pref_channel == "USSD" or row.get("preferred_channel_ussd", 0) == 1:
            return "USSD"
        elif row.get("clv_tier", "Bronze") in ["Platinum", "Gold"]:
            return "OUTBOUND_CALL"
        else:
            return "SMS"

    def _estimate_response_rate(self, propensity_score: float) -> float:
        """Estimate response rate based on propensity score."""
        if propensity_score > 0.8:
            return 0.35
        elif propensity_score > 0.65:
            return 0.20
        elif propensity_score > 0.5:
            return 0.10
        else:
            return 0.03

    def generate(self, campaign_type: str, max_size: int = 5000) -> pd.DataFrame:
        """Dispatch to the correct campaign builder based on campaign_type."""
        logger.info(f"Generating opportunity base for campaign type: {campaign_type}...")
        
        if campaign_type == "churn_retention":
            df_out = self._churn_retention_base(max_size)
        elif campaign_type == "bundle_upsell":
            df_out = self._bundle_upsell_base(max_size)
        elif campaign_type == "voice_topup":
            df_out = self._voice_topup_base(max_size)
        elif campaign_type == "reactivation":
            df_out = self._reactivation_base(max_size)
        elif campaign_type == "loyalty_reward":
            df_out = self._loyalty_reward_base(max_size)
        else:
            raise ValueError(f"Unknown campaign type: {campaign_type}")
            
        # Ensure returned dataframe is sorted by priority_score descending
        if len(df_out) > 0:
            df_out = df_out.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
            
        logger.info(f"Generated {len(df_out)} opportunities for {campaign_type}")
        return df_out

    def _churn_retention_base(self, max_size: int) -> pd.DataFrame:
        """Generate Churn Retention campaign opportunity base."""
        df = self.df.copy()
        
        # Filters
        mask = (
            (df["churn_risk_score"] >= settings.CHURN_RISK_THRESHOLD) &
            (~df["customer_id"].isin(self.suppressed_ids))
        )
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        # Numeric CLV Tier mapping
        clv_map = {"Platinum": 1.0, "Gold": 0.75, "Silver": 0.5, "Bronze": 0.25}
        clv_numeric = filtered_df["clv_tier"].map(clv_map).fillna(0.25)
        
        # Tenure normalized
        max_tenure = filtered_df["tenure_months"].max()
        tenure_norm = filtered_df["tenure_months"] / (max_tenure if max_tenure > 0 else 1.0)
        
        # Priority score
        filtered_df["priority_score"] = (filtered_df["churn_risk_score"] * 0.5) + (clv_numeric * 0.3) + (tenure_norm * 0.2)
        
        # Sort and limit
        filtered_df = filtered_df.sort_values(by="priority_score", ascending=False).head(max_size)
        
        # Construct CampaignOpportunity records
        opportunities = []
        budget_map = {"Platinum": "Premium", "Gold": "Standard", "Silver": "Basic", "Bronze": "Minimal"}
        
        for _, row in filtered_df.iterrows():
            clv = row.get("clv_tier", "Bronze")
            if clv in ["Platinum", "Gold"]:
                offer = "Premium Retention Bundle"
            elif clv == "Silver":
                offer = "Loyalty Discount"
            else:
                offer = "Standard Retention SMS"
                
            channel = self._assign_channel(row)
            score = float(row.get("churn_risk_score", 0.5))
            
            opportunities.append(CampaignOpportunity(
                customer_id=row["customer_id"],
                campaign_name="churn_retention",
                segment_label=row.get("ml_segment", "Standard"),
                rfm_segment=row.get("rfm_segment", "Standard"),
                churn_risk_score=score,
                propensity_score=score,  # For churn, risk score serves as propensity
                clv_tier=clv,
                recommended_offer=offer,
                recommended_channel=channel,
                priority_score=float(row["priority_score"]),
                estimated_response_rate=self._estimate_response_rate(score),
                retention_budget=budget_map.get(clv, "Minimal")
            ))
            
        return pd.DataFrame([asdict(o) for o in opportunities])

    def _bundle_upsell_base(self, max_size: int) -> pd.DataFrame:
        """Generate Bundle Upsell campaign opportunity base."""
        df = self.df.copy()
        
        # Filters
        mask = (
            (df["propensity_bundle"] >= settings.PROPENSITY_THRESHOLD) &
            (df["churn_risk_score"] < 0.5) &
            (~df["customer_id"].isin(self.suppressed_ids))
        )
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        clv_map = {"Platinum": 1.0, "Gold": 0.75, "Silver": 0.5, "Bronze": 0.25}
        clv_numeric = filtered_df["clv_tier"].map(clv_map).fillna(0.25)
        
        # Priority score
        filtered_df["priority_score"] = (filtered_df["propensity_bundle"] * 0.7) + (clv_numeric * 0.3)
        
        # Sort and limit
        filtered_df = filtered_df.sort_values(by="priority_score", ascending=False).head(max_size)
        
        opportunities = []
        budget_map = {"Platinum": "Premium", "Gold": "Standard", "Silver": "Basic", "Bronze": "Minimal"}
        median_data = filtered_df["data_usage_mb_30d"].median() if "data_usage_mb_30d" in filtered_df.columns else 1000
        
        for _, row in filtered_df.iterrows():
            clv = row.get("clv_tier", "Bronze")
            offer = "1GB Daily Bundle" if row.get("data_usage_mb_30d", 0) < median_data else "5GB Monthly Bundle"
            channel = self._assign_channel(row)
            score = float(row.get("propensity_bundle", 0.5))
            
            opportunities.append(CampaignOpportunity(
                customer_id=row["customer_id"],
                campaign_name="bundle_upsell",
                segment_label=row.get("ml_segment", "Standard"),
                rfm_segment=row.get("rfm_segment", "Standard"),
                churn_risk_score=float(row.get("churn_risk_score", 0.0)),
                propensity_score=score,
                clv_tier=clv,
                recommended_offer=offer,
                recommended_channel=channel,
                priority_score=float(row["priority_score"]),
                estimated_response_rate=self._estimate_response_rate(score),
                retention_budget=budget_map.get(clv, "Minimal")
            ))
            
        return pd.DataFrame([asdict(o) for o in opportunities])

    def _voice_topup_base(self, max_size: int) -> pd.DataFrame:
        """Generate Voice Top-up campaign opportunity base."""
        df = self.df.copy()
        
        # Filters
        mask = (
            (df["propensity_topup"] >= settings.PROPENSITY_THRESHOLD) &
            (df["churn_risk_score"] < 0.5) &
            (~df["customer_id"].isin(self.suppressed_ids))
        )
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        clv_map = {"Platinum": 1.0, "Gold": 0.75, "Silver": 0.5, "Bronze": 0.25}
        clv_numeric = filtered_df["clv_tier"].map(clv_map).fillna(0.25)
        
        # Priority score
        filtered_df["priority_score"] = (filtered_df["propensity_topup"] * 0.7) + (clv_numeric * 0.3)
        
        filtered_df = filtered_df.sort_values(by="priority_score", ascending=False).head(max_size)
        
        opportunities = []
        budget_map = {"Platinum": "Premium", "Gold": "Standard", "Silver": "Basic", "Bronze": "Minimal"}
        
        for _, row in filtered_df.iterrows():
            clv = row.get("clv_tier", "Bronze")
            offer = "Voice Top-up: Get ₦100 bonus on ₦500 recharge"
            channel = self._assign_channel(row)
            score = float(row.get("propensity_topup", 0.5))
            
            opportunities.append(CampaignOpportunity(
                customer_id=row["customer_id"],
                campaign_name="voice_topup",
                segment_label=row.get("ml_segment", "Standard"),
                rfm_segment=row.get("rfm_segment", "Standard"),
                churn_risk_score=float(row.get("churn_risk_score", 0.0)),
                propensity_score=score,
                clv_tier=clv,
                recommended_offer=offer,
                recommended_channel=channel,
                priority_score=float(row["priority_score"]),
                estimated_response_rate=self._estimate_response_rate(score),
                retention_budget=budget_map.get(clv, "Minimal")
            ))
            
        return pd.DataFrame([asdict(o) for o in opportunities])

    def _reactivation_base(self, max_size: int) -> pd.DataFrame:
        """Generate Customer Reactivation campaign opportunity base."""
        df = self.df.copy()
        
        q20_recharge = df["total_recharge_90d"].quantile(0.20)
        
        # Filters
        mask = (
            (df["days_since_last_recharge"] > 20) &
            (df["propensity_reactivation"] >= settings.PROPENSITY_THRESHOLD) &
            (df["total_recharge_90d"] > q20_recharge) &
            (~df["customer_id"].isin(self.suppressed_ids))
        )
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        max_recharge = filtered_df["total_recharge_90d"].max()
        recharge_norm = filtered_df["total_recharge_90d"] / (max_recharge if max_recharge > 0 else 1.0)
        
        # Priority score
        filtered_df["priority_score"] = (filtered_df["propensity_reactivation"] * 0.6) + (recharge_norm * 0.4)
        
        filtered_df = filtered_df.sort_values(by="priority_score", ascending=False).head(max_size)
        
        opportunities = []
        budget_map = {"Platinum": "Premium", "Gold": "Standard", "Silver": "Basic", "Bronze": "Minimal"}
        
        for _, row in filtered_df.iterrows():
            clv = row.get("clv_tier", "Bronze")
            offer = "Welcome Back: 2x data for 7 days"
            # Reactivation campaign uses USSD for dormant customers
            channel = "USSD"
            score = float(row.get("propensity_reactivation", 0.5))
            
            opportunities.append(CampaignOpportunity(
                customer_id=row["customer_id"],
                campaign_name="reactivation",
                segment_label=row.get("ml_segment", "Standard"),
                rfm_segment=row.get("rfm_segment", "Standard"),
                churn_risk_score=float(row.get("churn_risk_score", 0.0)),
                propensity_score=score,
                clv_tier=clv,
                recommended_offer=offer,
                recommended_channel=channel,
                priority_score=float(row["priority_score"]),
                estimated_response_rate=self._estimate_response_rate(score),
                retention_budget=budget_map.get(clv, "Minimal")
            ))
            
        return pd.DataFrame([asdict(o) for o in opportunities])

    def _loyalty_reward_base(self, max_size: int) -> pd.DataFrame:
        """Generate Loyalty Reward campaign opportunity base."""
        df = self.df.copy()
        
        # Filters
        mask = (
            (df["rfm_segment"].isin(["Champions", "Loyal Customers"])) &
            (df["churn_risk_score"] < 0.3) &
            (~df["customer_id"].isin(self.suppressed_ids))
        )
        filtered_df = df[mask].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        max_clv = filtered_df["clv_90d"].max()
        clv_norm = filtered_df["clv_90d"] / (max_clv if max_clv > 0 else 1.0)
        
        # Priority score CLV-weighted
        filtered_df["priority_score"] = clv_norm
        
        filtered_df = filtered_df.sort_values(by="priority_score", ascending=False).head(max_size)
        
        opportunities = []
        budget_map = {"Platinum": "Premium", "Gold": "Standard", "Silver": "Basic", "Bronze": "Minimal"}
        
        for _, row in filtered_df.iterrows():
            clv = row.get("clv_tier", "Bronze")
            offer = "Exclusive Bundle Discount" if clv in ["Platinum", "Gold"] else "Loyalty Points Bonus"
            channel = self._assign_channel(row)
            
            # Since loyalty has no direct propensity, estimate from CLV
            score = float(clv_norm.loc[row.name])
            
            opportunities.append(CampaignOpportunity(
                customer_id=row["customer_id"],
                campaign_name="loyalty_reward",
                segment_label=row.get("ml_segment", "Standard"),
                rfm_segment=row.get("rfm_segment", "Standard"),
                churn_risk_score=float(row.get("churn_risk_score", 0.0)),
                propensity_score=score,
                clv_tier=clv,
                recommended_offer=offer,
                recommended_channel=channel,
                priority_score=float(row["priority_score"]),
                estimated_response_rate=self._estimate_response_rate(score),
                retention_budget=budget_map.get(clv, "Minimal")
            ))
            
        return pd.DataFrame([asdict(o) for o in opportunities])

    def save_opportunity_base(self, df: pd.DataFrame, campaign_name: str) -> Path:
        """Save campaign opportunity base DataFrame to CSV."""
        date_today = datetime.today().strftime("%Y%m%d")
        file_path = settings.CAMPAIGNS_DIR / f"{campaign_name}_{date_today}.csv"
        
        if len(df) > 0:
            df.to_csv(file_path, index=False)
            
            # Calculate metrics for logging
            reach_channel = df["recommended_channel"].value_counts().to_dict()
            avg_propensity = df["propensity_score"].mean()
            logger.info(f"Saved campaign '{campaign_name}' opportunity base to {file_path}")
            logger.info(f"Targeted Customers: {len(df)}")
            logger.info(f"Reach by channel: {reach_channel}")
            logger.info(f"Avg Propensity: {avg_propensity:.4f}")
        else:
            # Create empty file with columns
            empty_df = pd.DataFrame(columns=[f.name for f in CampaignOpportunity.__dataclass_fields__.values()])
            empty_df.to_csv(file_path, index=False)
            logger.info(f"Generated empty opportunity base for '{campaign_name}' (0 targets). Saved to {file_path}")
            
        return file_path

    def generate_all(self, max_size: int = 5000) -> dict[str, Path]:
        """Loop over all campaign types and generate opportunity bases."""
        campaign_types = ["churn_retention", "bundle_upsell", "voice_topup", "reactivation", "loyalty_reward"]
        saved_paths = {}
        
        for ctype in campaign_types:
            opp_df = self.generate(ctype, max_size)
            path = self.save_opportunity_base(opp_df, ctype)
            saved_paths[ctype] = path
            
        return saved_paths

def generate_opportunity_bases(master_df: pd.DataFrame) -> dict[str, Path]:
    """Orchestrator function wrapper for train.py execution."""
    generator = OpportunityBaseGenerator(master_df)
    return generator.generate_all()
