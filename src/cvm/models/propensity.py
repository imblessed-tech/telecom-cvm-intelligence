from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.cvm.config import settings
from src.cvm.models.base_model import BaseCVMModel

logger = logging.getLogger(__name__)

@dataclass
class PropensityResult:
    customer_id: str
    offer_type: str
    propensity_score: float
    targetable: bool
    priority_rank: int

class PropensityModel(BaseCVMModel):
    def __init__(self, offer_type: str):
        super().__init__()
        if offer_type not in ["bundle_upgrade", "voice_topup", "reactivation"]:
            raise ValueError(f"Invalid offer type: {offer_type}")
        self.offer_type = offer_type
        self.threshold = settings.PROPENSITY_THRESHOLD

    def engineer_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer propensity labels from the data using business rules."""
        if self.offer_type == "bundle_upgrade":
            labels = (
                (df["high_data_user"] == 1) & 
                (df["data_usage_trend"] > 0) & 
                (df["offer_acceptance_rate"] > 0.15)
            ).astype(int).values
        elif self.offer_type == "voice_topup":
            median_recharge = df["recharge_count_30d"].median()
            labels = (
                (df["recharge_count_30d"] > median_recharge) & 
                (df["avg_recharge_amount"] < 500) & 
                (df["frequent_recharger"] == 1)
            ).astype(int).values
        elif self.offer_type == "reactivation":
            q25_recharge = df["total_recharge_90d"].quantile(0.25)
            labels = (
                (df["days_since_last_recharge"] > 20) & 
                (df["total_recharge_90d"] > q25_recharge)
            ).astype(int).values
        else:
            raise ValueError(f"Unknown offer type: {self.offer_type}")
            
        positive_rate = np.mean(labels)
        logger.info(f"Engineered propensity labels for {self.offer_type}: positive rate = {positive_rate:.2%}")
        return labels

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None, output_dir: Path = None) -> dict:
        """Train the propensity model."""
        self._feature_names = feature_names if feature_names is not None else []
        self._trained_at = datetime.now()
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, stratify=y, random_state=42
        )
        
        # Initialize model
        self._model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        self._model.fit(X_train, y_train)
        
        # Predict on test
        y_pred_proba = self._model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        logger.info(f"Propensity Model ({self.offer_type}) evaluation - ROC AUC: {auc_roc:.4f}, PR AUC: {auc_pr:.4f}")
        
        return {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        }

    def predict(self, X: np.ndarray, customer_ids: list[str] = None) -> list[PropensityResult]:
        """Predict propensity scores and rank targetable customers."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        probs = self._model.predict_proba(X)[:, 1]
        
        results = []
        n_customers = len(X)
        
        temp_list = []
        for i in range(n_customers):
            prob = float(probs[i])
            targetable = bool(prob >= self.threshold)
            cust_id = customer_ids[i] if customer_ids is not None else str(i)
            
            temp_list.append({
                "index": i,
                "customer_id": cust_id,
                "propensity_score": prob,
                "targetable": targetable
            })
            
        # Sort targetable indices by score descending to get priority_rank
        targetable_items = [item for item in temp_list if item["targetable"]]
        targetable_items.sort(key=lambda x: x["propensity_score"], reverse=True)
        
        # Map customer_id or index to its priority_rank (1-based)
        rank_mapping = {}
        for rank, item in enumerate(targetable_items, start=1):
            rank_mapping[item["index"]] = rank
            
        for i, item in enumerate(temp_list):
            priority_rank = rank_mapping.get(i, -1)  # Use -1 for non-targetable
            results.append(PropensityResult(
                customer_id=item["customer_id"],
                offer_type=self.offer_type,
                propensity_score=item["propensity_score"],
                targetable=item["targetable"],
                priority_rank=priority_rank
            ))
            
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for classes."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save model to path."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        super().save(path)

    def load(self, path: Path) -> None:
        """Load model from path."""
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
        logger.info(f"Model loaded from {path}")

    @property
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return self._model is not None
