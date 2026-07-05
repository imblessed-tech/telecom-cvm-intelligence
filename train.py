import os
import sys
from pathlib import Path

# [0] Add src/ to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

import logging
import numpy as np
import pandas as pd

# [0] Configure logging (StreamHandler + FileHandler to "training.log")
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "training.log", mode="w")
    ]
)
logger = logging.getLogger("train_pipeline")

from src.cvm.config import settings, ensure_directories
from src.cvm.data.loader import load_ibm, load_events
from src.cvm.data.generator import generate_events
from src.cvm.data.preprocessor import CVMPreprocessor
from src.cvm.features.rfm import RFMAnalyser
from src.cvm.features.behavioural import BehaviouralFeatureEngineer
from src.cvm.models.churn_predictor import ChurnPredictor
from src.cvm.models.segmentation import CustomerSegmentation
from src.cvm.models.propensity import PropensityModel
from src.cvm.models.clv_predictor import CLVPredictor

def main():
    logger.info("Initializing CVM Intelligence Platform training pipeline...")
    
    # [0] Call ensure_directories()
    ensure_directories()

    # [1] Load IBM churn dataset
    logger.info(f"Loading IBM churn dataset from {settings.TELCO_CHURN_FILE}...")
    ibm_df = load_ibm(settings.TELCO_CHURN_FILE)

    # [2] Generate customer events (if not already generated)
    if not settings.CUSTOMER_EVENTS_FILE.exists():
        logger.info("Customer events file not found. Generating events...")
        events_df = generate_events(ibm_df, settings.CUSTOMER_EVENTS_FILE)
    else:
        logger.info(f"Loading customer events from {settings.CUSTOMER_EVENTS_FILE}...")
        events_df = load_events(settings.CUSTOMER_EVENTS_FILE)

    # [3] Preprocess and join datasets
    logger.info("Preprocessing and joining datasets...")
    preprocessor = CVMPreprocessor()
    master_df = preprocessor.fit_transform(ibm_df, events_df)
    master_df.to_csv(settings.FEATURES_FILE, index=False)
    preprocessor.save(settings.PREPROCESSOR_FILE)

    # [4] Feature engineering
    logger.info("Performing feature engineering...")
    rfm_analyser = RFMAnalyser()
    master_df = rfm_analyser.compute(master_df)
    rfm_analyser.save_rfm_scores(master_df, settings.RFM_FILE)
    
    engineer = BehaviouralFeatureEngineer()
    master_df = engineer.engineer(master_df)

    # [5] Define feature matrix X
    logger.info("Defining feature matrix X...")
    non_feature_cols = [
        "customer_id", "churn_label", "rfm_segment", "lifecycle_stage", 
        "arpu_tier", "preferred_channel", "gender"
    ]
    drop_cols = [col for col in non_feature_cols if col in master_df.columns]
    feature_df = master_df.drop(columns=drop_cols)
    feature_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = master_df[feature_columns].values
    feature_names = feature_columns
    logger.info(f"Feature matrix shape: {X.shape}. Number of features: {len(feature_columns)}")

    # [6] Train Churn Predictor
    logger.info("Training Churn Predictor model...")
    y_churn = master_df["churn_label"].values
    churn_model = ChurnPredictor()
    churn_eval = churn_model.train(X, y_churn, feature_names, output_dir=settings.SHAP_DIR)
    
    # Add churn scores back to master_df
    churn_preds = churn_model.predict(X, customer_ids=master_df["customer_id"].tolist())
    master_df["churn_risk_score"] = [p.churn_probability for p in churn_preds]
    churn_model.save(settings.CHURN_MODEL_FILE)

    # [7] Train Customer Segmentation
    logger.info("Training Customer Segmentation model...")
    seg_model = CustomerSegmentation()
    seg_model.find_optimal_k(X, output_dir=settings.REPORT_DIR)
    seg_summary = seg_model.train(X, master_df, feature_names, settings.REPORT_DIR, k=settings.N_CLUSTERS)
    
    # Add segment labels to master_df
    seg_preds = seg_model.predict(X, customer_ids=master_df["customer_id"].tolist())
    master_df["ml_segment"] = [p.segment_label for p in seg_preds]
    seg_model.save(settings.SEGMENTATION_MODEL_FILE)

    # [8] Train three Propensity Models
    logger.info("Training Campaign Propensity Models...")
    for offer_type in ["bundle_upgrade", "voice_topup", "reactivation"]:
        logger.info(f"Training propensity model for offer type: {offer_type}...")
        model = PropensityModel(offer_type)
        y_prop = model.engineer_labels(master_df)
        model.train(X, y_prop, feature_names)
        
        # Add propensity scores to master_df
        prop_preds = model.predict(X, customer_ids=master_df["customer_id"].tolist())
        
        if offer_type == "bundle_upgrade":
            col_name = "propensity_bundle"
            save_path = settings.PROPENSITY_BUNDLE_FILE
        elif offer_type == "voice_topup":
            col_name = "propensity_topup"
            save_path = settings.PROPENSITY_TOPUP_FILE
        else:
            col_name = "propensity_reactivation"
            save_path = settings.PROPENSITY_REACTIVATION_FILE
            
        master_df[col_name] = [p.propensity_score for p in prop_preds]
        model.save(save_path)

    # [9] Train CLV Predictor
    logger.info("Training CLV Predictor model...")
    clv_model = CLVPredictor()
    y_clv_log = clv_model.engineer_target(master_df)
    y_raw = master_df["total_recharge_90d"].values
    clv_model.train(X, y_clv_log, y_raw, feature_names)
    
    # Add CLV predictions to master_df
    clv_preds = clv_model.predict(X, customer_ids=master_df["customer_id"].tolist())
    master_df["clv_90d"] = [p.predicted_clv_90d for p in clv_preds]
    master_df["clv_tier"] = [p.clv_tier for p in clv_preds]
    clv_model.save(settings.CLV_MODEL_FILE)

    # [10] Save enriched master_df (now has all scores)
    logger.info(f"Saving enriched master DataFrame to {settings.FEATURES_FILE}...")
    master_df.to_csv(settings.FEATURES_FILE, index=False)

    # [11] Generate campaign opportunity bases
    try:
        from src.cvm.campaign.opportunity_base import generate_opportunity_bases
        logger.info("Generating campaign opportunity bases...")
        generate_opportunity_bases(master_df)
    except ImportError:
        logger.warning("opportunity_base.py or generate_opportunity_bases not found. Skipping opportunity base generation.")

    # [12] Generate dashboard
    try:
        from src.cvm.visualization.dashboard import generate_dashboard
        logger.info("Generating CVM dashboard...")
        generate_dashboard(master_df)
    except ImportError:
        logger.warning("dashboard.py or generate_dashboard not found. Skipping dashboard generation.")

    # [13] Print final summary table
    logger.info("Training pipeline execution finished successfully.")
    print("\n" + "="*55)
    print("      CVM INTELLIGENCE PLATFORM - TRAINING SUMMARY")
    print("="*55)
    print(f"Total Customers Processed: {len(master_df)}")
    print("-"*55)
    
    # Churn metrics
    if "churn_risk_score" in master_df.columns:
        critical_count = (master_df["churn_risk_score"] > 0.8).sum()
        high_count = ((master_df["churn_risk_score"] <= 0.8) & (master_df["churn_risk_score"] > 0.65)).sum()
        print(f"Churn Risk - Critical (>0.8): {critical_count}")
        print(f"Churn Risk - High (0.65-0.8):  {high_count}")
        
    # Segmentation metrics
    if "ml_segment" in master_df.columns:
        print("-"*55)
        print("Customer Segments:")
        seg_counts = master_df["ml_segment"].value_counts()
        for seg, count in seg_counts.items():
            print(f"  * {seg}: {count}")
            
    # Propensity metrics
    print("-"*55)
    print("Average Campaign Propensity:")
    for prop_col in ["propensity_bundle", "propensity_topup", "propensity_reactivation"]:
        if prop_col in master_df.columns:
            print(f"  * {prop_col}: {master_df[prop_col].mean():.2%}")
            
    # CLV metrics
    if "clv_tier" in master_df.columns:
        print("-"*55)
        print("CLV Tier Distribution:")
        clv_counts = master_df["clv_tier"].value_counts()
        for tier, count in clv_counts.items():
            print(f"  * {tier}: {count}")
            
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
