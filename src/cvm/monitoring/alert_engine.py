import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src.cvm.config import settings
from src.cvm.monitoring.drift_detector import DriftResult

logger = logging.getLogger(__name__)

@dataclass
class CVMAlert:
    alert_id: str
    alert_type: str  # "DATA_DRIFT" / "PREDICTION_DRIFT" / "PERFORMANCE_DROP" / "MISSING_DATA"
    severity: str    # "WARNING" / "CRITICAL"
    feature_name: str
    metric_value: float
    threshold: float
    message: str
    recommended_action: str
    timestamp: str

class AlertEngine:
    def __init__(self):
        self.active_alerts: list[CVMAlert] = []

    def evaluate_drift(self, drift_results: list[DriftResult]) -> list[CVMAlert]:
        """Evaluate feature drift results and generate alerts."""
        new_alerts = []
        for r in drift_results:
            if not r.drift_detected and r.psi <= 0.1:
                continue
                
            if r.psi >= 0.2:
                severity = "CRITICAL"
                action = "Retrain all models immediately. Feature distribution has shifted significantly."
            else:
                severity = "WARNING"
                action = "Monitor closely. Schedule retraining within 2 weeks."
                
            alert = CVMAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="DATA_DRIFT",
                severity=severity,
                feature_name=r.feature_name,
                metric_value=r.psi,
                threshold=settings.DRIFT_PSI_THRESHOLD,
                message=f"Feature '{r.feature_name}' has drifted. PSI = {r.psi:.4f} (KS p-val = {r.ks_p_value:.4f})",
                recommended_action=action,
                timestamp=datetime.now().isoformat()
            )
            new_alerts.append(alert)
            self.active_alerts.append(alert)
            
        logger.warning(f"Feature drift evaluation completed. Generated {len(new_alerts)} new alerts.")
        return new_alerts

    def evaluate_prediction_drift(self, result: DriftResult) -> list[CVMAlert]:
        """Evaluate prediction score drift result and generate alerts."""
        new_alerts = []
        if result.drift_detected or result.psi > 0.1:
            severity = "CRITICAL" if result.psi >= 0.2 else "WARNING"
            alert = CVMAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="PREDICTION_DRIFT",
                severity=severity,
                feature_name="churn_risk_score",
                metric_value=result.psi,
                threshold=settings.DRIFT_PSI_THRESHOLD,
                message=f"Prediction score distribution has shifted. PSI = {result.psi:.4f} (KS p-val = {result.ks_p_value:.4f})",
                recommended_action="Investigate downstream: check if business conditions changed or data pipeline has issues.",
                timestamp=datetime.now().isoformat()
            )
            new_alerts.append(alert)
            self.active_alerts.append(alert)
            
        logger.warning(f"Prediction drift evaluation completed. Generated {len(new_alerts)} alerts.")
        return new_alerts

    def evaluate_performance(self, current_auc: float, reference_auc: float, model_name: str) -> list[CVMAlert]:
        """Evaluate model AUC performance drops and generate alerts."""
        new_alerts = []
        drop = reference_auc - current_auc
        
        if drop >= 0.02:
            severity = "CRITICAL" if drop >= settings.PERFORMANCE_DROP_THRESHOLD else "WARNING"
            alert = CVMAlert(
                alert_id=str(uuid.uuid4()),
                alert_type="PERFORMANCE_DROP",
                severity=severity,
                feature_name=model_name,
                metric_value=drop,
                threshold=settings.PERFORMANCE_DROP_THRESHOLD,
                message=f"Model performance drop detected on {model_name}. Current AUC: {current_auc:.4f} (Ref: {reference_auc:.4f}, drop: {drop:.2%})",
                recommended_action="Model accuracy below acceptable threshold. Prioritise retraining.",
                timestamp=datetime.now().isoformat()
            )
            new_alerts.append(alert)
            self.active_alerts.append(alert)
            
        logger.warning(f"Performance evaluation completed. Generated {len(new_alerts)} alerts.")
        return new_alerts

    def get_summary(self) -> dict:
        """Get summary statistics of active alerts."""
        summary = {
            "severity_counts": {"WARNING": 0, "CRITICAL": 0},
            "type_counts": {"DATA_DRIFT": 0, "PREDICTION_DRIFT": 0, "PERFORMANCE_DROP": 0, "MISSING_DATA": 0},
            "latest_timestamp": None
        }
        for a in self.active_alerts:
            summary["severity_counts"][a.severity] = summary["severity_counts"].get(a.severity, 0) + 1
            summary["type_counts"][a.alert_type] = summary["type_counts"].get(a.alert_type, 0) + 1
            if summary["latest_timestamp"] is None or a.timestamp > summary["latest_timestamp"]:
                summary["latest_timestamp"] = a.timestamp
                
        return summary

    def save_alert_log(self, path: Path) -> None:
        """Save alert engine logs to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized_alerts = [asdict(a) for a in self.active_alerts]
        with open(path, "w") as f:
            json.dump(serialized_alerts, f, indent=4)
        logger.info(f"Saved {len(self.active_alerts)} alerts to alert log file: {path}")
