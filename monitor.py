import sys
from pathlib import Path

# Add src/ to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cvm_monitor")

from src.cvm.config import settings, ensure_directories
from src.cvm.data.preprocessor import CVMPreprocessor
from src.cvm.models.churn_predictor import ChurnPredictor
from src.cvm.models.segmentation import CustomerSegmentation
from src.cvm.models.propensity import PropensityModel
from src.cvm.models.clv_predictor import CLVPredictor
from src.cvm.monitoring.drift_detector import DriftDetector, DriftResult
from src.cvm.monitoring.alert_engine import AlertEngine, CVMAlert

def generate_html_report(drift_results: list[DriftResult], pred_drift: DriftResult, active_alerts: list[CVMAlert], summary: dict, output_path: Path):
    """Generate a rich aesthetic HTML report for CVM monitoring."""
    
    # Rows for feature drift table
    feature_rows_html = ""
    for r in drift_results:
        # Determine background color based on severity
        if r.severity == "severe":
            bg_color = "rgba(239, 68, 68, 0.15)"
            text_color = "#ef4444"
            badge = '<span class="badge badge-danger">CRITICAL</span>'
        elif r.severity == "moderate":
            bg_color = "rgba(245, 158, 11, 0.15)"
            text_color = "#f59e0b"
            badge = '<span class="badge badge-warning">WARNING</span>'
        else:
            bg_color = "rgba(16, 185, 129, 0.15)"
            text_color = "#10b981"
            badge = '<span class="badge badge-success">STABLE</span>'
            
        feature_rows_html += f"""
        <tr style="background-color: {bg_color};">
            <td style="font-weight: 600;">{r.feature_name}</td>
            <td style="font-family: monospace;">{r.psi:.4f}</td>
            <td style="font-family: monospace;">{r.ks_statistic:.4f}</td>
            <td style="font-family: monospace;">{r.ks_p_value:.4e}</td>
            <td style="color: {text_color}; font-weight: bold;">{r.mean_shift_pct:+.2%}</td>
            <td>{badge}</td>
        </tr>
        """
        
    # Rows for alert table
    alert_rows_html = ""
    if not active_alerts:
        alert_rows_html = "<tr><td colspan='5' style='text-align: center; color: #888;'>No active alerts raised.</td></tr>"
    else:
        for a in active_alerts:
            badge = '<span class="badge badge-danger">CRITICAL</span>' if a.severity == "CRITICAL" else '<span class="badge badge-warning">WARNING</span>'
            alert_rows_html += f"""
            <tr>
                <td>{badge}</td>
                <td style="font-weight: 600;">{a.alert_type}</td>
                <td>{a.feature_name}</td>
                <td>{a.message}</td>
                <td style="font-size: 0.9em; color: #cbd5e1;">{a.recommended_action}</td>
            </tr>
            """
            
    # Estimated timestamps
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CVM Intelligence - Model Drift & Alert Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --accent-blue: #3b82f6;
            --accent-teal: #0d9488;
            --border-color: #334155;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3a8a, #0d9488);
            padding: 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
        }}
        
        .header::after {{
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.05), transparent 50%);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }}
        
        .header p {{
            color: #cbd5e1;
            font-size: 1.1rem;
        }}
        
        .grid-summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
        }}
        
        .card-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }}
        
        .card-value {{
            font-size: 2rem;
            font-weight: 800;
            color: #f8fafc;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin: 2.5rem 0 1rem 0;
            border-left: 4px solid var(--accent-blue);
            padding-left: 0.75rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        
        th, td {{
            padding: 1rem 1.25rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: rgba(15, 23, 42, 0.6);
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        
        .badge-success {{ background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }}
        .badge-warning {{ background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }}
        .badge-danger {{ background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}
        
        .pred-drift-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .pred-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        
        .pred-metric {{
            background-color: rgba(15, 23, 42, 0.4);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CVM Data Drift & Stability Report</h1>
        <p>Generated on: {report_time} | Environment: Local Monitoring Pipeline</p>
    </div>

    <div class="grid-summary">
        <div class="card">
            <div class="card-title">Features Monitored</div>
            <div class="card-value">{len(drift_results)}</div>
        </div>
        <div class="card">
            <div class="card-title">Drifted Features</div>
            <div class="card-value" style="color: #ef4444;">{sum(1 for r in drift_results if r.drift_detected)}</div>
        </div>
        <div class="card">
            <div class="card-title">Critical Alerts</div>
            <div class="card-value" style="color: #ef4444;">{summary["severity_counts"].get("CRITICAL", 0)}</div>
        </div>
        <div class="card">
            <div class="card-title">Active Warnings</div>
            <div class="card-value" style="color: #f59e0b;">{summary["severity_counts"].get("WARNING", 0)}</div>
        </div>
    </div>

    <div class="section-title">Prediction Score Stability</div>
    <div class="pred-drift-card">
        <h3>Output Feature: churn_risk_score</h3>
        <p style="color: #94a3b8; margin-bottom: 1rem;">Monitoring the output distribution shifts of the XGBoost classifier predictions.</p>
        <div class="pred-grid">
            <div class="pred-metric">
                <div class="card-title">PSI Score</div>
                <div class="card-value" style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem;">{pred_drift.psi:.4f}</div>
            </div>
            <div class="pred-metric">
                <div class="card-title">KS statistic</div>
                <div class="card-value" style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem;">{pred_drift.ks_statistic:.4f}</div>
            </div>
            <div class="pred-metric">
                <div class="card-title">KS p-value</div>
                <div class="card-value" style="font-family: 'JetBrains Mono', monospace; font-size: 1.5rem;">{pred_drift.ks_p_value:.4e}</div>
            </div>
            <div class="pred-metric">
                <div class="card-title">Drift Status</div>
                <div style="margin-top: 0.5rem;">
                    {"<span class='badge badge-danger'>DRIFT DETECTED</span>" if pred_drift.drift_detected else "<span class='badge badge-success'>STABLE</span>"}
                </div>
            </div>
        </div>
    </div>

    <div class="section-title">Feature Drift Analysis Details</div>
    <table>
        <thead>
            <tr>
                <th>Feature Name</th>
                <th>PSI Metric</th>
                <th>KS Statistic</th>
                <th>KS p-value</th>
                <th>Mean Shift %</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {feature_rows_html}
        </tbody>
    </table>

    <div class="section-title">Active Alert Logs</div>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Alert Type</th>
                <th>Source Feature</th>
                <th>Description</th>
                <th>Recommended Action</th>
            </tr>
        </thead>
        <tbody>
            {alert_rows_html}
        </tbody>
    </table>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Dashboard HTML report written to {output_path}")

def main():
    logger.info("Initializing CVM model monitoring script...")
    ensure_directories()

    # [1] Load all trained models and preprocessor
    logger.info("Loading preprocessor and trained models...")
    preprocessor = CVMPreprocessor.load(settings.PREPROCESSOR_FILE)
    
    churn_model = ChurnPredictor()
    churn_model.load(settings.CHURN_MODEL_FILE)
    
    # [2] Load the master features DataFrame (training reference)
    logger.info(f"Loading master feature store from {settings.FEATURES_FILE}...")
    master_df = pd.read_csv(settings.FEATURES_FILE)

    # [3] Set reference distributions in DriftDetector using training data
    non_feature_cols = [
        "customer_id", "churn_label", "rfm_segment", "lifecycle_stage", 
        "arpu_tier", "preferred_channel", "gender", "churn_risk_score", 
        "segment_label", "ml_segment", "propensity_bundle", "propensity_topup", 
        "propensity_reactivation", "clv_90d", "clv_tier"
    ]
    feature_columns = [col for col in master_df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(master_df[col])]
    
    detector = DriftDetector()
    detector.set_reference(master_df, feature_columns)

    # [4] Simulate "current" data (take a 20% sample and inject noise)
    logger.info("Simulating new batch of customer data...")
    current_df = master_df.sample(frac=0.2, random_state=42).copy()
    
    # Add noise to 4 specific columns to simulate drift
    np.random.seed(settings.RANDOM_STATE)
    drift_features = ["arpu", "data_usage_mb_30d", "days_since_last_recharge", "avg_recharge_amount"]
    drift_features = [col for col in drift_features if col in current_df.columns]
    
    for col in drift_features:
        scale_factor = np.random.uniform(1.2, 1.45)
        current_df[col] = current_df[col] * scale_factor
        logger.info(f"Simulated data drift on '{col}' (scaled by {scale_factor:.2f})")

    # [5] Run drift detection on current data
    logger.info("Detecting data drift...")
    drift_results = detector.detect_drift(current_df, feature_columns)
    
    # Run prediction drift on Churn Predictor scores
    X_ref = master_df[feature_columns].values
    X_curr = current_df[feature_columns].values
    
    ref_scores = churn_model.predict_proba(X_ref)[:, 1]
    curr_scores = churn_model.predict_proba(X_curr)[:, 1]
    
    pred_drift_result = detector.detect_prediction_drift(ref_scores, curr_scores)

    # [6] Run alert engine on drift results
    logger.info("Evaluating alert thresholds...")
    engine = AlertEngine()
    data_drift_alerts = engine.evaluate_drift(drift_results)
    pred_drift_alerts = engine.evaluate_prediction_drift(pred_drift_result)
    
    # Simulate a performance degradation alert to test engine
    simulated_perf_alerts = engine.evaluate_performance(
        current_auc=0.74, 
        reference_auc=0.81, 
        model_name="XGBClassifier"
    )
    
    all_alerts = data_drift_alerts + pred_drift_alerts + simulated_perf_alerts
    summary = engine.get_summary()

    # [7] Generate monitoring dashboard HTML report
    report_path = settings.MONITORING_DIR / "drift_report.html"
    generate_html_report(drift_results, pred_drift_result, all_alerts, summary, report_path)

    # [8] Save alert log to JSON
    log_path = settings.MONITORING_DIR / "alerts_log.json"
    engine.save_alert_log(log_path)

    # [9] Print summary to console
    print("\n" + "="*55)
    print("      CVM INTELLIGENCE PLATFORM - MONITORING SUMMARY")
    print("="*55)
    print(f"Total Features Monitored: {len(drift_results)}")
    print(f"Features with Drift:      {sum(1 for r in drift_results if r.drift_detected)}")
    print(f"Prediction Drift Status:  {'DRIFT DETECTED' if pred_drift_result.drift_detected else 'STABLE'} (PSI = {pred_drift_result.psi:.4f})")
    print("-"*55)
    print("Active Alerts Summary:")
    print(f"  * Critical Alerts:      {summary['severity_counts'].get('CRITICAL', 0)}")
    print(f"  * Warning Alerts:       {summary['severity_counts'].get('WARNING', 0)}")
    print(f"  * Performance Alerts:   {summary['type_counts'].get('PERFORMANCE_DROP', 0)}")
    print("-"*55)
    print(f"Alert Log Path: {log_path}")
    print(f"Drift Report:   {report_path}")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
