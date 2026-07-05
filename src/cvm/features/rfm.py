import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RFMResult:
    customer_id: str
    recency_days: int
    frequency_30d: int
    monetary_30d: float
    r_score: int
    f_score: int
    m_score: int
    rfm_score: int
    rfm_segment: str

class RFMAnalyser:
    def __init__(self):
        """No special initialisation needed"""
        pass

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM metrics for each customer.
        """
        logger.info("Computing RFM metrics")
        required_cols = ["customer_id", "days_since_last_recharge", "recharge_count_30d", "total_recharge_30d"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        df = df.copy()
        df = df.sort_values("customer_id").reset_index(drop=True)

        # Bin recency (lower days = higher score)
        R = pd.qcut(
            df["days_since_last_recharge"].rank(method='first'), 
            q=5, 
            labels=[5,4,3,2,1], 
            duplicates="drop"
        ).astype(int)

        # Bin frequency (higher count = higher score)
        F = pd.qcut(
            df["recharge_count_30d"].rank(method="first"), 
            q=5, 
            labels=[1,2,3,4,5], 
            duplicates="drop"
        ).astype(int)

        # Bin monetary value (higher amount = higher score)
        M = pd.qcut(
            df["total_recharge_30d"].rank(method="first"), 
            q=5, 
            labels=[1,2,3,4,5], 
            duplicates="drop"
        ).astype(int)

        df["r_score"] = R
        df["f_score"] = F
        df["m_score"] = M
        df["rfm_score"] = R*100 + F*10 + M

        for idx, row in df.iterrows():
            rfm_segment = self._assign_segment(row["rfm_score"])
            df.loc[idx, "rfm_segment"] = rfm_segment
        
        return df

    def _assign_segment(self, rfm_score: int) -> str:
        """Assign segment label based on RFM score."""
        # Ensure scores are split cleanly
        r = rfm_score // 100
        f = (rfm_score // 10) % 10
        m = rfm_score % 10

        # 1. Champions (Highest Recency, Frequency, and Monetary)
        if r >= 5 and f >= 4 and m >= 4:
            return 'Champions'    
        # 2. Lost Customers (Haven't purchased in the longest time, regardless of others)
        elif r == 1:
            return 'Lost Customers'    
        # 3. New Customers (Just joined, lowest frequency)
        elif r == 5 and f == 1:
            return 'New Customers'    
        # 4. At Risk (Used to be good, now going quiet)
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'    
        # 5. Loyal Customers (Buy regularly, good value)
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'    
        # 6. Potential Loyalists (Recent but not yet frequent)
        elif r >= 4 and f <= 3:
            return 'Potential Loyalists'    
        # 7. Hibernating (Long gone, low value)
        elif r <= 2 and f <= 2:
            return 'Hibernating'    
        # 8. Catch-all fallback
        else:
            return 'Hibernating'
    
    def get_segment_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group by segment and compute metrics"""
        segment_group = df.groupby("rfm_segment").agg({
            "days_since_last_recharge": ["mean"],
            "recharge_count_30d": ["mean"],
            "total_recharge_30d": ["mean"],
            "monthly_charges": ["mean"],
        })
        return segment_group

    def save_rfm_scores(self, df: pd.DataFrame, path: Path) -> None:
        """Save the customer_id and RFM scores to a CSV file."""
        df.to_csv(path, columns=["customer_id", "rfm_score"], index=False)
        logger.info(f"Saved RFM scores to {path}")
