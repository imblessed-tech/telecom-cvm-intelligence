from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cvm.models.base_model import BaseCVMModel

logger = logging.getLogger(__name__)

@dataclass
class CLVPrediction:
    customer_id: str
    predicted_clv_90d: float
    clv_tier: str
    retention_budget_tier: str
    monthly_arpu_estimate: float

class CLVPredictor(BaseCVMModel):
    def __init__(self):
        super().__init__()
        self._clv_percentiles = {}

    def engineer_target(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer CLV target values and apply log transform."""
        clv_90d = df["total_recharge_90d"] * (1 - df["churn_label"] * 0.8)
        # Apply np.log1p transform
        return np.log1p(clv_90d).values

    def train(self, X: np.ndarray, y_log: np.ndarray, y_raw: np.ndarray, feature_names: list = None) -> dict:
        """Train the CLV prediction model."""
        self._feature_names = feature_names if feature_names is not None else []
        self._trained_at = datetime.now()
        
        # Initialize model
        self._model = GradientBoostingRegressor(
            n_estimators=200, 
            max_depth=4, 
            learning_rate=0.05, 
            random_state=42
        )
        
        # Perform 5-fold CV on log target
        logger.info("Performing 5-fold cross-validation...")
        cv_r2 = cross_val_score(self._model, X, y_log, cv=5, scoring="r2")
        cv_r2_mean = float(np.mean(cv_r2))
        
        # Fit model on the full dataset
        logger.info("Fitting model on full dataset...")
        self._model.fit(X, y_log)
        
        # Predict on full dataset to calculate training metrics
        y_pred_log = self._model.predict(X)
        y_pred = np.expm1(y_pred_log)
        
        mae = float(mean_absolute_error(y_raw, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_raw, y_pred)))
        r2 = float(r2_score(y_raw, y_pred))
        
        # Store training percentile thresholds from y_raw (p50, p75, p90)
        self._clv_percentiles = {
            "p50": float(np.percentile(y_raw, 50)),
            "p75": float(np.percentile(y_raw, 75)),
            "p90": float(np.percentile(y_raw, 90))
        }
        
        logger.info(f"CLVPredictor training metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, CV R2 Mean: {cv_r2_mean:.4f}")
        logger.info(f"CLV percentiles stored: {self._clv_percentiles}")
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_r2_mean": cv_r2_mean
        }

    def predict(self, X: np.ndarray, customer_ids: list[str] = None) -> list[CLVPrediction]:
        """Predict 90-day CLV and assign value/budget tiers."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        y_pred_log = self._model.predict(X)
        y_pred = np.expm1(y_pred_log)
        
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0.0)
        
        results = []
        n_customers = len(X)
        
        for i in range(n_customers):
            pred_clv = float(y_pred[i])
            cust_id = customer_ids[i] if customer_ids is not None else str(i)
            
            # Assign CLV tier and budget based on stored training percentiles
            p50 = self._clv_percentiles.get("p50", 0.0)
            p75 = self._clv_percentiles.get("p75", 0.0)
            p90 = self._clv_percentiles.get("p90", 0.0)
            
            if pred_clv > p90:
                tier = "Platinum"
                budget = "Premium"
            elif pred_clv > p75:
                tier = "Gold"
                budget = "Standard"
            elif pred_clv > p50:
                tier = "Silver"
                budget = "Basic"
            else:
                tier = "Bronze"
                budget = "Minimal"
                
            results.append(CLVPrediction(
                customer_id=cust_id,
                predicted_clv_90d=pred_clv,
                clv_tier=tier,
                retention_budget_tier=budget,
                monthly_arpu_estimate=pred_clv / 3.0
            ))
            
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (not supported for regression)."""
        raise AttributeError("CLVPredictor is a regressor and does not output probabilities")

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
