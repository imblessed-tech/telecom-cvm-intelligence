from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier
import shap
from src.cvm.config import settings
from src.cvm.models.base_model import BaseCVMModel

logger = logging.getLogger(__name__)

@dataclass
class ChurnPrediction:
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    risk_tier: str
    top_risk_factors: list[str]
    recommended_action: str

@dataclass
class ChurnEvaluation:
    auc_roc: float
    auc_pr: float
    f1_score: float
    precision: float
    recall: float
    optimal_threshold: float

class ChurnPredictor(BaseCVMModel):
    def __init__(self):
        super().__init__()
        self.shap_explainer = None
        self.optimal_threshold = 0.5
        self._class_weights = {}

    def train(self, 
                X: np.ndarray,
                y: np.ndarray,
                feature_names: list[str],
                output_dir: Path = None
                ) -> ChurnEvaluation:
        """Train the churn prediction model"""
        self._feature_names = feature_names
        self._trained_at = datetime.now()
        
        # 1. Compute the class weights from training labels
        self._class_weights = {
            "n_neg": (y == 0).sum(),
            "n_pos": (y == 1).sum(),
            "scale_pos_weight": (y == 0).sum() / (y == 1).sum()
        }
        logger.info(f"Class weights: {self._class_weights}")

        # 2. Split data into training and testing sets
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, range(len(y)), test_size=settings.TEST_SIZE, stratify=y, random_state=42
        )
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Testing set shape: {X_test.shape}")

        # 3. Train the XGBoost model
        self._model = XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        scale_pos_weight=self._class_weights["scale_pos_weight"],
                        random_state=42,
                        eval_metric="auc",
                        early_stopping_rounds=20,
                    )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # 4. Find optimal threshold
        best_threshold = 0.5
        best_f1 = 0

        thresholds = np.arange(0.3, 0.85, 0.05)
        logger.info("Finding optimal threshold between 0.3 and 0.8 in steps of 0.05...")

        y_pred_proba = self._model.predict_proba(X_test)[:, 1]
        logger.info(f"Shape of predicted probabilities: {y_pred_proba.shape}")

        for thresh in thresholds:
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_test, y_pred_binary)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        self.optimal_threshold = best_threshold
        logger.info(f"Optimal threshold: {self.optimal_threshold} (F1-score: {best_f1:.4f})")

        # 5. Evaluate model on the test set using the optimal threshold
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        y_pred_binary = (y_pred_proba >= self.optimal_threshold).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)

        # 6. Fit SHAP explainer
        logger.info("Fitting SHAP explainer...")
        self.shap_explainer = shap.TreeExplainer(self._model)
        logger.info("SHAP explainer fitted")

        evaluation = ChurnEvaluation(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            f1_score=f1,
            precision=precision,
            recall=recall,
            optimal_threshold=self.optimal_threshold
        )

        logger.info(f"Model evaluation: {evaluation}")


        # 6. Save evaluation results and SHAP plots
        if output_dir:
            self._save_shap_summary_plot(X_test, feature_names, output_dir)

        return evaluation

    def _save_shap_summary_plot(self, X_test: np.ndarray, feature_names: list[str], output_dir: Path) -> None:
        """Save SHAP summary plot."""
        # Compute SHAP values
        logger.info("Computing SHAP feature importance...")
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self._model)
        
        self.shap_values = self.shap_explainer.shap_values(X_test)
        
        # Handle list of matrices (for binary classifier)
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        summary_plot_path = output_dir / "churn_shap_summary.png"
        plt.savefig(summary_plot_path)
        plt.close()
        logger.info(f"Saved SHAP summary plot to {summary_plot_path}")

    def predict(self, X: np.ndarray, customer_ids: list[str] = None) -> list[ChurnPrediction]:
        """Predict churn for customers."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        probs = self._model.predict_proba(X)[:, 1]
        
        shap_values = self.shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            if shap_values.shape[0] == 2:
                shap_values = shap_values[1]
            elif shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
                
        predictions = []
        n_customers = len(X)
        
        for i in range(n_customers):
            prob = float(probs[i])
            pred_binary = bool(prob >= self.optimal_threshold)
            
            # Risk tier and recommended action mapping
            if prob > 0.8:
                risk_tier = "Critical"
                rec_action = "Immediate outreach: personal call + premium retention offer"
            elif prob > 0.65:
                risk_tier = "High"
                rec_action = "Priority SMS: personalised discount bundle offer"
            elif prob > 0.4:
                risk_tier = "Medium"
                rec_action = "Automated campaign: loyalty reward notification"
            else:
                risk_tier = "Low"
                rec_action = "Standard nurture: regular engagement campaign"
                
            # Get top 3 risk factors by absolute SHAP value
            shap_row = shap_values[i]
            abs_shap = np.abs(shap_row)
            top_3_indices = np.argsort(abs_shap)[-3:][::-1]
            
            top_features = []
            for idx in top_3_indices:
                if idx < len(self._feature_names):
                    top_features.append(self._feature_names[idx])
                else:
                    top_features.append(f"feature_{idx}")
                    
            cust_id = customer_ids[i] if customer_ids is not None else str(i)
            
            predictions.append(ChurnPrediction(
                customer_id=cust_id,
                churn_probability=prob,
                churn_predicted=pred_binary,
                risk_tier=risk_tier,
                top_risk_factors=top_features,
                recommended_action=rec_action
            ))
            
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the model."""
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

        
    