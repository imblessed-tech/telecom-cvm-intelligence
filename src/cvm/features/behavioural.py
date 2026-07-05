import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BehaviouralFeatureEngineer:

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute behavioural features"""
        df = df.copy()
        df = df.sort_values("customer_id").reset_index(drop=True)

        df = self._compute_arpu(df)
        df = self._compute_value_tier(df)
        df = self._compute_engagement_score(df)
        df = self._compute_churn_risk_signals(df)
        df = self._compute_upsell_signals(df)
        df = self._compute_lifecycle_stage(df)

        return df
    
    def _compute_arpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Average Revenue Per User (ARPU)"""
        df = df.copy()
        df["arpu"] = df["total_recharge_90d"] / 3
        return df
    
    def _compute_value_tier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute customer value tier based on ARPU"""
        df = df.copy()
        if len(df) < 4:
            df["arpu_tier"] = "Bronze"
        else:
            df["arpu_tier"] = pd.qcut(df["arpu"].rank(method='first'), q=4, labels=["Bronze", "Silver", "Gold", "Platinum"])
        return df
        
    def _compute_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute engagement score"""
        df = df.copy()
        df['engagement_score'] = (df['active_days_90d'] / 90) * 0.4 + (df['app_usage_rate']) * 0.3 + (df['offer_acceptance_rate']) * 0.3
        return df

    def _compute_churn_risk_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute binary risk signal columns for churn model features."""
        df = df.copy()
            
        # 1. Declining recharge (declining by more than ₦50 per day)
        df["declining_recharge"] = np.where(df["recharge_trend"] < -50, 1, 0)

        # 2. Long silence (no recharge activity for more than 14 days)
        df["long_silence"] = np.where(df["days_since_last_recharge"] > 14, 1, 0)

        # 3. Low activity (active less than 20 days out of the last 90)
        df["low_activity"] = np.where(df["active_days_90d"] < 20, 1, 0)

        # 4. Recharge drop flag (current month frequency < 50% of 90-day average)
        drop_condition = df["recharge_count_30d"] < (df["recharge_count_90d"] / 6)
        df["recharge_drop_flag"] = np.where(drop_condition, 1, 0)

        return df

    def _compute_upsell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute binary upsell signal columns for propensity models."""
        df = df.copy()

        # 1. High data user (top 25% of users over the last 30 days)
        data_75th = df["data_usage_mb_30d"].quantile(0.75)
        df["high_data_user"] = np.where(
            df["data_usage_mb_30d"] > data_75th, 1, 0
        )

        # 2. Frequent recharger (above median recharge count over the last 30 days)
        recharge_median = df["recharge_count_30d"].median()
        df["frequent_recharger"] = np.where(
            df["recharge_count_30d"] > recharge_median, 1, 0
        )

        # 3. High offer responder (acceptance rate strictly greater than 20%)
        df["high_offer_responder"] = np.where(
            df["offer_acceptance_rate"] > 0.20, 1, 0
        )

        # 4. App user (app usage rate strictly greater than 30%)
        df["app_user"] = np.where(df["app_usage_rate"] > 0.3, 1, 0)

        # 5. Premium customer (uses .isin() to safely check for 'Gold' or 'Platinum')
        premium_condition = df["arpu_tier"].isin(["Gold", "Platinum"])
        df["premium_customer"] = np.where(premium_condition, 1, 0)

        return df
    
    def _compute_lifecycle_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign each customer to a lifecycle stage based on tenure and activity."""
        df = df.copy()

        # Pre-calculate 75th percentile for ARPU
        arpu_75th = df["arpu"].quantile(0.75)

        # Define conditions in order of priority (most specific first)
        conditions = [
            # 1. High Value Loyal (Long tenure, top tier spending)
            (df["tenure_months"] >= 12) & (df["arpu"] > arpu_75th),
            # 2. Churned (Longer absence takes precedence over Lapsing)
            (df["tenure_months"] >= 3) & (df["days_since_last_recharge"] > 60),
            # 3. Lapsing
            (df["tenure_months"] >= 3) & (df["days_since_last_recharge"] > 30),
            # 4. New Active
            (df["tenure_months"] < 3) & (df["active_days_90d"] > 30),
            # 5. New At-Risk
            (df["tenure_months"] < 3) & (df["active_days_90d"] < 30),
        ]

        # Map corresponding labels to each condition
        choices = [
            "High Value Loyal",
            "Churned",
            "Lapsing",
            "New Active",
            "New At-Risk",
        ]

        # Execute selection with "Active" as the fallback/default category
        df["lifecycle_stage"] = np.select(conditions, choices, default="Active")

        return df

        