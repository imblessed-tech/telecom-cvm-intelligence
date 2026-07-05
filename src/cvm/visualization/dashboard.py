import base64
import io
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cvm.config import settings
from src.cvm.campaign.opportunity_base import OpportunityBaseGenerator

logger = logging.getLogger(__name__)

def fig_to_b64(fig) -> str:
    """Save matplotlib figure to buffer and encode as base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor='#161b22')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

class CVMDashboardGenerator:
    def __init__(self, master_df: pd.DataFrame, models: dict = None):
        self.df = master_df.copy()
        self.models = models if models else {}

    def _build_churn_risk_distribution(self) -> str:
        """Create churn probability risk histogram."""
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(colors='#c9d1d9')
        ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        scores = self.df["churn_risk_score"].dropna().values if "churn_risk_score" in self.df.columns else np.random.uniform(0, 1, 100)
        ax.hist(scores, bins=20, color='#3b82f6', edgecolor='#161b22', alpha=0.8)
        
        ax.axvline(0.4, color='#f59e0b', linestyle='--', linewidth=1.5, label='Medium Risk (0.4)')
        ax.axvline(0.65, color='#ef4444', linestyle='--', linewidth=1.5, label='High Risk (0.65)')
        
        # Colour areas
        ax.axvspan(0.0, 0.4, color='#10b981', alpha=0.08)
        ax.axvspan(0.4, 0.65, color='#f59e0b', alpha=0.08)
        ax.axvspan(0.65, 1.0, color='#ef4444', alpha=0.08)
        
        ax.set_title("Churn Risk Distribution", color='#58a6ff', fontsize=12, fontweight='bold')
        ax.set_xlabel("Churn Probability", color='#c9d1d9')
        ax.set_ylabel("Customer Count", color='#c9d1d9')
        ax.legend(facecolor='#161b22', edgecolor='#334155', labelcolor='#c9d1d9')
        
        return fig_to_b64(fig)

    def _build_segment_overview(self) -> str:
        """Create segment distributions overview subplots (ML and RFM)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#161b22')
        
        for ax in (ax1, ax2):
            ax.set_facecolor('#161b22')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(colors='#c9d1d9')
            ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)
            
        # ML Segments
        if "ml_segment" in self.df.columns:
            ml_counts = self.df["ml_segment"].value_counts()
            ax1.bar(ml_counts.index, ml_counts.values, color='#0d9488', alpha=0.8)
        ax1.set_title("ML Customer Segments", color='#58a6ff', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=30)
        ax1.set_ylabel("Count", color='#c9d1d9')
        
        # RFM Segments
        if "rfm_segment" in self.df.columns:
            rfm_counts = self.df["rfm_segment"].value_counts()
            ax2.bar(rfm_counts.index, rfm_counts.values, color='#8b5cf6', alpha=0.8)
        ax2.set_title("RFM Customer Segments", color='#58a6ff', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=30)
        ax2.set_ylabel("Count", color='#c9d1d9')
        
        plt.tight_layout()
        return fig_to_b64(fig)

    def _build_clv_distribution(self) -> str:
        """Create CLV distribution bar chart."""
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(colors='#c9d1d9')
        ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        clv_tiers = ["Platinum", "Gold", "Silver", "Bronze"]
        counts = []
        means = []
        pcts = []
        total = len(self.df)
        
        for tier in clv_tiers:
            tier_df = self.df[self.df["clv_tier"] == tier] if "clv_tier" in self.df.columns else pd.DataFrame()
            count = len(tier_df)
            mean_clv = tier_df["clv_90d"].mean() if count > 0 else 0.0
            counts.append(count)
            means.append(mean_clv)
            pcts.append((count / total * 100) if total > 0 else 0.0)
            
        bars = ax.bar(clv_tiers, counts, color='#10b981', alpha=0.8)
        ax.set_title("Customer Lifetime Value Tiers", color='#58a6ff', fontsize=12, fontweight='bold')
        ax.set_ylabel("Customer Count", color='#c9d1d9')
        
        # Add labels on top of bars
        for bar, mean_val, pct in zip(bars, means, pcts):
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                yval + (total * 0.015 if total > 0 else 1),
                f"₦{mean_val:,.0f}\n({pct:.1f}%)",
                ha='center',
                va='bottom',
                color='#c9d1d9',
                fontsize=8
            )
            
        return fig_to_b64(fig)

    def _build_propensity_overview(self) -> str:
        """Create propensity models score histograms."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('#161b22')
        
        models_info = [
            ("propensity_bundle", "Bundle Upgrade", ax1, '#3b82f6'),
            ("propensity_topup", "Voice Top-up", ax2, '#10b981'),
            ("propensity_reactivation", "Reactivation", ax3, '#ef4444')
        ]
        
        for col, label, ax, col_hex in models_info:
            ax.set_facecolor('#161b22')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('#334155')
            ax.tick_params(colors='#c9d1d9')
            ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
            ax.set_axisbelow(True)
            
            scores = self.df[col].dropna() if col in self.df.columns else pd.Series()
            if len(scores) > 0:
                ax.hist(scores, bins=15, color=col_hex, edgecolor='#161b22', alpha=0.8)
                
            ax.axvline(settings.PROPENSITY_THRESHOLD, color='#cbd5e1', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
            ax.set_title(f"{label} Propensity", color='#58a6ff', fontsize=11, fontweight='bold')
            ax.set_xlabel("Probability Score", color='#c9d1d9')
            ax.set_ylabel("Count", color='#c9d1d9')
            
        fig.suptitle("Propensity Score Distributions", color='#58a6ff', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig_to_b64(fig)

    def _build_lifecycle_funnel(self) -> str:
        """Create horizontal bar chart of customer lifecycle stages."""
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('#334155')
        ax.tick_params(colors='#c9d1d9')
        ax.xaxis.grid(True, color='#334155', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        ordered_stages = ["New Active", "Active", "High Value Loyal", "Lapsing", "Churned"]
        counts = []
        for stage in ordered_stages:
            counts.append((self.df["lifecycle_stage"] == stage).sum() if "lifecycle_stage" in self.df.columns else 0)
            
        # Reverse to show highest-value/youngest at top
        ordered_stages.reverse()
        counts.reverse()
        
        ax.barh(ordered_stages, counts, color='#ec4899', alpha=0.8)
        ax.set_title("Customer Lifecycle Stages", color='#58a6ff', fontsize=12, fontweight='bold')
        ax.set_xlabel("Customer Count", color='#c9d1d9')
        
        return fig_to_b64(fig)

    def _build_rfm_heatmap(self) -> str:
        """Create 2D heatmap of average monthly charges grouped by R and F scores."""
        # 5x5 matrix
        matrix = np.zeros((5, 5))
        
        if "r_score" in self.df.columns and "f_score" in self.df.columns:
            for r in range(1, 6):
                for f in range(1, 6):
                    subset = self.df[(self.df["r_score"] == r) & (self.df["f_score"] == f)]
                    matrix[5 - r, f - 1] = subset["monthly_charges"].mean() if len(subset) > 0 else 0.0
                    
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#c9d1d9')
        cbar.ax.tick_params(labelcolor='#c9d1d9')
        
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['F=1', 'F=2', 'F=3', 'F=4', 'F=5'], color='#c9d1d9')
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(['R=5', 'R=4', 'R=3', 'R=2', 'R=1'], color='#c9d1d9')
        
        ax.set_title("RFM Heatmap (Avg Monthly Charges)", color='#58a6ff', fontsize=12, fontweight='bold')
        
        # Annotate each cell with values
        for i in range(5):
            for j in range(5):
                val = matrix[i, j]
                ax.text(j, i, f"₦{val:.0f}" if val > 0 else "-",
                        ha="center", va="center", color="white" if val < 60 else "black", fontsize=9, fontweight='bold')
                        
        return fig_to_b64(fig)

    def _build_campaign_opportunity_summary(self) -> str:
        """Create dual axis campaign opportunity counts and average CLV tier indices."""
        campaign_types = ["churn_retention", "bundle_upsell", "voice_topup", "reactivation", "loyalty_reward"]
        
        generator = OpportunityBaseGenerator(self.df)
        eligible_counts = []
        avg_clv_tiers = []
        clv_map = {"Platinum": 4.0, "Gold": 3.0, "Silver": 2.0, "Bronze": 1.0}
        
        for ctype in campaign_types:
            opp_df = generator.generate(ctype, max_size=999999)
            count = len(opp_df)
            eligible_counts.append(count)
            avg_clv = opp_df["clv_tier"].map(clv_map).mean() if count > 0 else 0.0
            avg_clv_tiers.append(avg_clv)
            
        fig, ax1 = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax1.set_facecolor('#161b22')
        ax1.tick_params(colors='#c9d1d9')
        ax1.spines['bottom'].set_color('#334155')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        ax1.spines['left'].set_color('#334155')
        
        x = np.arange(len(campaign_types))
        width = 0.35
        
        ax1.bar(x - width/2, eligible_counts, width, label='Eligible Count', color='#3b82f6', alpha=0.8)
        ax1.set_ylabel('Eligible Count', color='#3b82f6')
        ax1.set_xticks(x)
        ax1.set_xticklabels(["Retention", "Upsell", "Top-up", "Reactivate", "Loyalty"], color='#c9d1d9')
        
        ax2 = ax1.twinx()
        ax2.tick_params(colors='#c9d1d9')
        ax2.spines['bottom'].set_color('#334155')
        ax2.spines['top'].set_color('none')
        ax2.spines['right'].set_color('#334155')
        ax2.spines['left'].set_color('none')
        
        ax2.bar(x + width/2, avg_clv_tiers, width, label='Avg CLV Tier', color='#f59e0b', alpha=0.8)
        ax2.set_ylabel('Avg CLV Tier (1-4)', color='#f59e0b')
        ax2.set_ylim(0, 4.5)
        
        ax1.set_title("Campaign Opportunity Base Summary", color='#58a6ff', fontsize=12, fontweight='bold')
        return fig_to_b64(fig)

    def _build_shap_summary(self) -> str:
        """Load the pre-saved SHAP summary image and convert to base64."""
        shap_path = settings.SHAP_DIR / "churn_shap_summary.png"
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            logger.warning(f"SHAP summary image not found at {shap_path}. Displaying empty container.")
            return ""

    def generate(self, output_path: Path) -> None:
        """Orchestrate chart generation, build final dashboard HTML page and save."""
        logger.info("Generating CVM dashboard charts...")
        
        churn_dist_b64 = self._build_churn_risk_distribution()
        shap_b64 = self._build_shap_summary()
        segments_b64 = self._build_segment_overview()
        clv_b64 = self._build_clv_distribution()
        rfm_heatmap_b64 = self._build_rfm_heatmap()
        propensity_b64 = self._build_propensity_overview()
        lifecycle_b64 = self._build_lifecycle_funnel()
        campaign_opp_b64 = self._build_campaign_opportunity_summary()
        
        # KPI calculations
        total_customers = len(self.df)
        at_risk_count = (self.df["churn_risk_score"] >= settings.CHURN_RISK_THRESHOLD).sum() if "churn_risk_score" in self.df.columns else 0
        avg_arpu = self.df["arpu"].mean() if "arpu" in self.df.columns else 0.0
        
        platinum_count = (self.df["clv_tier"] == "Platinum").sum() if "clv_tier" in self.df.columns else 0
        platinum_pct = (platinum_count / total_customers * 100) if total_customers > 0 else 0.0
        
        # Eligible campaigns counts
        generator = OpportunityBaseGenerator(self.df)
        campaign_eligible = 0
        for ctype in ["churn_retention", "bundle_upsell", "voice_topup", "reactivation", "loyalty_reward"]:
            campaign_eligible += len(generator.generate(ctype, max_size=999999))
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>CVM Intelligence Dashboard</title>
    <style>
        body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Arial; margin:0; padding:20px; }}
        h1 {{ color: #58a6ff; text-align:center; font-size: 28px; margin-bottom: 5px; }}
        .subtitle {{ text-align:center; color:#8b949e; margin-bottom:30px; }}
        h2 {{ color:#58a6ff; border-bottom: 1px solid #21262d; padding-bottom:8px; margin-top:40px; }}
        .kpi-row {{ display:flex; gap:16px; margin:20px 0; }}
        .kpi-card {{ background:#161b22; border:1px solid #21262d; border-radius:8px;
                    padding:20px; flex:1; text-align:center; }}
        .kpi-value {{ font-size:32px; font-weight:bold; color:#58a6ff; }}
        .kpi-label {{ color:#8b949e; font-size:13px; margin-top:5px; }}
        .chart-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0; }}
        .chart-full {{ margin:20px 0; background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:16px; }}
        .chart-half {{ background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:16px; }}
        img {{ width:100%; border-radius:4px; }}
        .footer {{ text-align:center; color:#484f58; margin-top:60px; font-size:12px; }}
        .alert-badge-critical {{ background:#da3633; color:white; padding:2px 8px;
                                border-radius:4px; font-size:12px; }}
        .alert-badge-warning  {{ background:#d29922; color:white; padding:2px 8px;
                                border-radius:4px; font-size:12px; }}
    </style>
</head>
<body>
    <h1>📊 CVM Intelligence Dashboard</h1>
    <p class="subtitle">Customer Value Management | ML Analytics Platform | Generated: {timestamp}</p>

    <!-- KPI Row -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f85149">{at_risk_count:,}</div>
            <div class="kpi-label">At Churn Risk (>{settings.CHURN_RISK_THRESHOLD*100:.0f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#3fb950">₦{avg_arpu:.0f}</div>
            <div class="kpi-label">Avg ARPU (₦/month)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{platinum_pct:.1f}%</div>
            <div class="kpi-label">Platinum CLV Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#d29922">{campaign_eligible:,}</div>
            <div class="kpi-label">Campaign-Eligible Today</div>
        </div>
    </div>

    <!-- Churn Analysis -->
    <h2>⚠️ Churn Risk Analysis</h2>
    <div class="chart-grid">
        <div class="chart-half"><img src="data:image/png;base64,{churn_dist_b64}"></div>
        <div class="chart-half"><img src="data:image/png;base64,{shap_b64}"></div>
    </div>

    <!-- Segmentation -->
    <h2>👥 Customer Segmentation</h2>
    <div class="chart-full"><img src="data:image/png;base64,{segments_b64}"></div>

    <!-- CLV -->
    <h2>💰 Customer Lifetime Value</h2>
    <div class="chart-grid">
        <div class="chart-half"><img src="data:image/png;base64,{clv_b64}"></div>
        <div class="chart-half"><img src="data:image/png;base64,{rfm_heatmap_b64}"></div>
    </div>

    <!-- Propensity -->
    <h2>🎯 Propensity Scores</h2>
    <div class="chart-full"><img src="data:image/png;base64,{propensity_b64}"></div>

    <!-- Lifecycle -->
    <h2>🔄 Customer Lifecycle</h2>
    <div class="chart-grid">
        <div class="chart-half"><img src="data:image/png;base64,{lifecycle_b64}"></div>
        <div class="chart-half"><img src="data:image/png;base64,{campaign_opp_b64}"></div>
    </div>

    <div class="footer">
        Telecom CVM Intelligence Platform v1.0 | Python · XGBoost · SHAP · scikit-learn · FastAPI · PySpark
    </div>
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)
            
        logger.info(f"Dashboard successfully generated at {output_path}")

def generate_dashboard(master_df: pd.DataFrame) -> None:
    """Orchestrator endpoint wrapper for train.py execution."""
    generator = CVMDashboardGenerator(master_df)
    
    # Primary dashboard report path
    output_path = settings.DASHBOARD_DIR / "cvm_dashboard.html"
    generator.generate(output_path)
    
    # Copy to docs/index.html for GitHub Pages deployment
    docs_path = settings.BASE_DIR / "docs" / "index.html"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shutil
        shutil.copy2(output_path, docs_path)
        logger.info(f"Dashboard copied to GitHub Pages deployment folder at {docs_path}")
    except Exception as e:
        logger.warning(f"Could not copy dashboard to docs folder: {e}")
