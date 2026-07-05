from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Optional

from src.cvm.config import settings
from src.cvm.models.base_model import BaseCVMModel

logger = logging.getLogger(__name__)

@dataclass
class SegmentResult:
    customer_id: str
    cluster_id: int
    segment_label: str
    segment_description: str
    recommended_campaign: str

class CustomerSegmentation(BaseCVMModel):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.cluster_profiles = pd.DataFrame()
        self._label_mapping = {}
        self._campaign_mapping = {} 

    def find_optimal_k(
            self, 
            X: np.ndarray, 
            k_range: range = range(3, 10), 
            output_dir: Path = None
        ) -> dict:
        """
        Test K values from 3 to 10.
        Plots elbow curve and silhouette scores.
        Returns the optimal K.
        """
        inertias = []
        silhouettes = []

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            
            inertias.append(model.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            logger.info(f"  K={k}: inertia={model.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

        if output_dir:
            self._plot_k_selection(
                list(k_range), inertias, silhouettes, output_dir
            )
        
        optimal_k = list(k_range)[np.argmax(silhouettes)]
        logger.info(f"Optimal K by silhouette score: {optimal_k}")
        return {"optimal_k": optimal_k, "silhouettes": silhouettes, "inertias": inertias}

    def _plot_k_selection(self, k_values, inertias, silhouettes, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(k_values, inertias, "bo-")
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia (lower = tighter clusters)")
        ax1.set_title("Elbow Method")
        ax1.grid(True)

        ax2.plot(k_values, silhouettes, "ro-")
        ax2.set_xlabel("Number of Clusters (K)")
        ax2.set_ylabel("Silhouette Score (higher = better)")
        ax2.set_title("Silhouette Score")
        ax2.grid(True)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "k_selection.png", dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"K-selection plot saved")
    
    def train(self, 
            X: np.ndarray, 
            df: pd.DataFrame, 
            feature_names: list = None, 
            output_dir: Path = None,
            k: Optional[int] = None,
            n_init: int = 10,
            random_state: int = 42
        ) -> dict:
        """Fit K-Means and profile each cluster."""
        n_clusters = k or settings.N_CLUSTERS
        logger.info(f"Training K-Means with K={n_clusters}...")
        
        self._feature_names = feature_names if feature_names is not None else []
        self._trained_at = datetime.now()
        
        # Scale input features
        scaled_X = self.scaler.fit_transform(X)
        
        # Train KMeans
        self._model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels = self._model.fit_predict(scaled_X)
        
        df_copy = df.copy()
        df_copy["cluster"] = labels
        
        # Profile clusters
        self._profile_clusters(df_copy)
        
        # Map cluster IDs to business labels based on profiles
        self._assign_labels()
        
        # Add labels to dataframe
        df_copy["segment_label"] = df_copy["cluster"].map(self._label_mapping)
        
        if output_dir:
            self._save_cluster_plot(scaled_X, labels, output_dir)
            
        silhouette = silhouette_score(scaled_X, labels)
        logger.info(f"K-Means training complete. Silhouette score: {silhouette:.4f}")

        return {
            "cluster_profiles": self.cluster_profiles,
            "silhouette_score": silhouette,
            "label_mapping": self._label_mapping
        }

    def _profile_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes mean feature values for each cluster."""
        possible_columns = [
            "arpu", "recharge_count_30d", "days_since_last_recharge", 
            "data_usage_mb_30d", "tenure_months", "churn_risk_score", "engagement_score"
        ]
        required_columns = [col for col in possible_columns if col in df.columns]
        
        if not required_columns:
            required_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if "cluster" in required_columns:
                required_columns.remove("cluster")

        cluster_profiles = df.groupby("cluster")[required_columns].mean()
        cluster_profiles["count"] = df.groupby("cluster").size()

        self.cluster_profiles = cluster_profiles
        logger.info(f"Cluster profiles computed:\n{cluster_profiles.to_string()}")
        return cluster_profiles

    def _assign_labels(self) -> None:
        """Assign segment labels based on cluster profiles using simple heuristics."""
        if self.cluster_profiles.empty:
            logger.warning("Cluster profiles are empty. Cannot assign labels.")
            return

        profiles = self.cluster_profiles.copy()
        n_clusters = len(profiles)
        
        self._label_mapping = {}
        self._campaign_mapping = {}
        
        # If we don't have exactly 6 clusters, map them to default segment labels
        if n_clusters != 6:
            for cid in profiles.index:
                label = f"Segment {cid}"
                self._label_mapping[int(cid)] = label
                self._campaign_mapping[int(cid)] = {
                    "label": label,
                    "description": "Custom customer segment based on behavioral clustering.",
                    "recommended_campaign": "Standard nurture: regular engagement campaign"
                }
            return
            
        # Standard 6-cluster business heuristic mapping
        dormant_cluster = profiles["days_since_last_recharge"].idxmax()
        
        profiles_remaining = profiles.drop(index=dormant_cluster)
        high_value_active_cluster = profiles_remaining["arpu"].idxmax()
        
        profiles_remaining = profiles_remaining.drop(index=high_value_active_cluster)
        risk_col = "churn_risk_score" if "churn_risk_score" in profiles_remaining.columns else "days_since_last_recharge"
        at_risk_cluster = profiles_remaining[risk_col].idxmax()
        
        profiles_remaining = profiles_remaining.drop(index=at_risk_cluster)
        loyal_declining_cluster = profiles_remaining["tenure_months"].idxmax()
        
        profiles_remaining = profiles_remaining.drop(index=loyal_declining_cluster)
        price_sensitive_cluster = profiles_remaining["recharge_count_30d"].idxmax()
        
        profiles_remaining = profiles_remaining.drop(index=price_sensitive_cluster)
        rising_stars_cluster = profiles_remaining.index[0]
        
        labels_dict = {
            high_value_active_cluster: (
                "High Value Active", 
                "High spend, frequent recharges, and low churn risk.",
                "Loyalty program: premium rewards & exclusive benefits"
            ),
            rising_stars_cluster: (
                "Rising Stars", 
                "Newer customers with high data usage and growth potential.",
                "Upsell: data bundle packages and gaming/streaming services"
            ),
            loyal_declining_cluster: (
                "Loyal But Declining", 
                "Long-term customers showing signs of declining activity.",
                "Retention campaign: targeted value offers & feedback survey"
            ),
            price_sensitive_cluster: (
                "Price Sensitive", 
                "Frequent recharges of small amounts, low overall spend.",
                "Cross-sell: micro-bundles & low-cost weekly offers"
            ),
            dormant_cluster: (
                "Dormant", 
                "No activity for a long time, low overall value.",
                "Win-back: massive discount on next recharge or reactivate bonus"
            ),
            at_risk_cluster: (
                "At Risk Heavyweights", 
                "Previously high spenders now showing strong signs of churn.",
                "Immediate outreach: high-value retention offer & proactive call"
            )
        }
        
        for cluster_id, (label, desc, campaign) in labels_dict.items():
            self._label_mapping[int(cluster_id)] = label
            self._campaign_mapping[int(cluster_id)] = {
                "label": label,
                "description": desc,
                "recommended_campaign": campaign
            }
            
        logger.info(f"Assigned labels mapping: {self._label_mapping}")

    def _save_cluster_plot(self, X_scaled: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
        """Save a 2D PCA cluster plot."""
        logger.info("Generating PCA cluster plot...")
        pca_X = self.pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = (labels == label)
            cluster_name = self._label_mapping.get(int(label), f"Segment {label}")
            plt.scatter(
                pca_X[mask, 0], 
                pca_X[mask, 1], 
                label=cluster_name,
                alpha=0.7,
                edgecolors="w",
                s=50
            )
            
        plt.title("Customer Segments Visualization (PCA 2D Projection)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "segment_clusters.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved cluster plot to {plot_path}")

    def predict(self, X: np.ndarray, customer_ids: list[str] = None) -> list[SegmentResult]:
        """Predict customer segments."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        scaled_X = self.scaler.transform(X)
        labels = self._model.predict(scaled_X)
        
        results = []
        n_customers = len(X)
        
        for i in range(n_customers):
            cluster_id = int(labels[i])
            campaign_info = self._campaign_mapping.get(
                cluster_id, 
                {
                    "label": f"Segment {cluster_id}",
                    "description": "Standard customer segment",
                    "recommended_campaign": "Standard nurture: regular engagement campaign"
                }
            )
            
            cust_id = customer_ids[i] if customer_ids is not None else str(i)
            
            results.append(SegmentResult(
                customer_id=cust_id,
                cluster_id=cluster_id,
                segment_label=campaign_info["label"],
                segment_description=campaign_info["description"],
                recommended_campaign=campaign_info["recommended_campaign"]
            ))
            
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict soft cluster assignments (probabilities based on distance)."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        scaled_X = self.scaler.transform(X)
        distances = self._model.transform(scaled_X)
        inv_distances = 1.0 / (distances + 1e-5)
        probs = inv_distances / inv_distances.sum(axis=1, keepdims=True)
        return probs

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
        



        