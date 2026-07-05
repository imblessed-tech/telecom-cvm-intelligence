import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.cvm.config import settings

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_detected: bool
    severity: str
    reference_mean: float
    current_mean: float
    mean_shift_pct: float

class DriftDetector:
    def __init__(self):
        self.reference_distributions = {}

    def set_reference(self, df: pd.DataFrame, feature_columns: list[str]) -> None:
        """Store reference distributions from training data."""
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Feature '{col}' not in reference DataFrame. Skipping...")
                continue
            
            # Store a sample of values (max 2000 rows), mean, std
            sample_size = min(len(df), 2000)
            sample_vals = df[col].dropna().sample(n=sample_size, random_state=settings.RANDOM_STATE).values
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            
            self.reference_distributions[col] = {
                "values": sample_vals,
                "mean": mean_val,
                "std": std_val
            }
        logger.info(f"Reference distributions stored for {len(self.reference_distributions)} features.")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Compute the Population Stability Index (PSI) between expected and actual distributions."""
        # Handle cases with constant or empty expected arrays
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        min_val, max_val = expected.min(), expected.max()
        if min_val == max_val:
            bin_edges = np.array([min_val - 0.1, min_val + 0.1])
        else:
            bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Calculate proportions in bins
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        expected_pcts = expected_counts / len(expected)
        
        actual_clipped = np.clip(actual, bin_edges[0], bin_edges[-1])
        actual_counts, _ = np.histogram(actual_clipped, bins=bin_edges)
        actual_pcts = actual_counts / len(actual)
        
        # Add epsilon to prevent log(0)
        expected_pcts = expected_pcts + 1e-4
        actual_pcts = actual_pcts + 1e-4
        
        # Re-normalize
        expected_pcts = expected_pcts / expected_pcts.sum()
        actual_pcts = actual_pcts / actual_pcts.sum()
        
        psi_val = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
        return float(psi_val)

    def detect_drift(self, current_df: pd.DataFrame, feature_columns: list[str]) -> list[DriftResult]:
        """Detect drift on a list of features comparing current_df against the reference."""
        results = []
        for col in feature_columns:
            if col not in self.reference_distributions:
                continue
                
            expected = self.reference_distributions[col]["values"]
            actual = current_df[col].dropna().values
            if len(actual) == 0:
                continue
                
            psi = self.compute_psi(expected, actual)
            
            # KS test
            ks_stat, p_val = ks_2samp(expected, actual)
            
            # Drift flag
            drift_detected = bool(psi > settings.DRIFT_PSI_THRESHOLD or p_val < settings.DRIFT_KS_ALPHA)
            
            # Severity mapping
            if psi >= 0.2:
                severity = "severe"
            elif psi >= 0.1:
                severity = "moderate"
            else:
                severity = "none"
                
            ref_mean = float(self.reference_distributions[col]["mean"])
            curr_mean = float(actual.mean())
            mean_shift = float((curr_mean - ref_mean) / (ref_mean + 1e-9))
            
            results.append(DriftResult(
                feature_name=col,
                psi=psi,
                ks_statistic=float(ks_stat),
                ks_p_value=float(p_val),
                drift_detected=drift_detected,
                severity=severity,
                reference_mean=ref_mean,
                current_mean=curr_mean,
                mean_shift_pct=mean_shift
            ))
            
        drifted_count = sum(1 for r in results if r.drift_detected)
        worst_psi = max((r.psi for r in results), default=0.0)
        logger.warning(f"Drift detection complete. Drifted features: {drifted_count}/{len(results)}. Worst PSI: {worst_psi:.4f}")
        
        return results

    def detect_prediction_drift(self, reference_scores: np.ndarray, current_scores: np.ndarray) -> DriftResult:
        """Detect PSI + KS drift on model prediction scores."""
        psi = self.compute_psi(reference_scores, current_scores)
        ks_stat, p_val = ks_2samp(reference_scores, current_scores)
        
        drift_detected = bool(psi > settings.DRIFT_PSI_THRESHOLD or p_val < settings.DRIFT_KS_ALPHA)
        
        if psi >= 0.2:
            severity = "severe"
        elif psi >= 0.1:
            severity = "moderate"
        else:
            severity = "none"
            
        ref_mean = float(reference_scores.mean())
        curr_mean = float(current_scores.mean())
        mean_shift = float((curr_mean - ref_mean) / (ref_mean + 1e-9))
        
        return DriftResult(
            feature_name="prediction_scores",
            psi=psi,
            ks_statistic=float(ks_stat),
            ks_p_value=float(p_val),
            drift_detected=drift_detected,
            severity=severity,
            reference_mean=ref_mean,
            current_mean=curr_mean,
            mean_shift_pct=mean_shift
        )
