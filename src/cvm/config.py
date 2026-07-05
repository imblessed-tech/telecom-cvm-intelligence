from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CAMPAIGNS_DIR: Path = DATA_DIR / "campaigns" / "opportunity_bases"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORT_DIR: Path = BASE_DIR / "reports"
    SHAP_DIR: Path = REPORT_DIR / "shap"
    MONITORING_DIR: Path = REPORT_DIR / "monitoring"
    DASHBOARD_DIR: Path = REPORT_DIR / "dashboards"

    # Raw data files
    TELCO_CHURN_FILE: Path = RAW_DIR / "telco_churn.csv"
    CUSTOMER_EVENTS_FILE: Path = RAW_DIR / "customer_events.csv"
    
    # Processed data files
    FEATURES_FILE: Path = PROCESSED_DIR / "features.csv"
    RFM_FILE: Path = PROCESSED_DIR / "rfm_scores.csv"

    # Model files
    CHURN_MODEL_FILE: Path = MODELS_DIR /"churn_model.joblib"
    SEGMENTATION_MODEL_FILE: Path = MODELS_DIR /"segmentation_model.joblib"
    PROPENSITY_BUNDLE_FILE: Path = MODELS_DIR /"propensity_bundle.joblib"
    PROPENSITY_TOPUP_FILE: Path = MODELS_DIR /"propensity_topup.joblib"
    PROPENSITY_REACTIVATION_FILE: Path = MODELS_DIR /"propensity_reactivation.joblib"
    CLV_MODEL_FILE: Path = MODELS_DIR /"clv_model.joblib"
    PREPROCESSOR_FILE: Path = MODELS_DIR /"preprocessor.joblib"

    # ML Hyperparameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_CLUSTERS: int = 6  # six customer segments
    CHURN_THRESHOLD_DAYS: int = 30  # no recharge for 30 days = churned
    CLV_HORIZON_DAYS: int = 90  # predict 90-day revenue
    ANOMALY_CONTAMINATION: float = 0.05

    # Business Rules
    HIGH_CLV_PERCENTILE: float = 0.75   # top 25% = high value
    CHURN_RISK_THRESHOLD: float = 0.65  # score above this = at risk
    PROPENSITY_THRESHOLD: float = 0.50  # score above this = targetable

    # Data Generation
    NORMAL_PHASE_DAYS: int = 60       # days in normal phase
    DECLINE_PHASE_DAYS: int = 75      # days in decline phase
    

    # Monitoring thresholds
    DRIFT_PSI_THRESHOLD: float = 0.2  # PSI above 0.2 = significant drift
    DRIFT_KS_ALPHA: float = 0.05  # KS test significance level
    PERFORMANCE_DROP_THRESHOLD: float = 0.05  # 5% AUC drop triggers alert#   

    # ── API settings 
    API_TITLE: str = "CVM Intelligence Platform"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
            env_file = ".env"         
            extra = "ignore"

settings = Settings()

def ensure_directories():
    """Ensure all required directories exist."""
    data_dirs = [
        settings.DATA_DIR,
        settings.RAW_DIR,
        settings.PROCESSED_DIR,
        settings.CAMPAIGNS_DIR,
        settings.MODELS_DIR,
        settings.REPORT_DIR,
        settings.SHAP_DIR,
        settings.MONITORING_DIR,
        settings.DASHBOARD_DIR,
    ]

    for directory in data_dirs:
        directory.mkdir(parents=True, exist_ok=True)