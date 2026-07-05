import logging
import time
from fastapi import HTTPException
import pandas as pd

from src.cvm.config import settings
from src.cvm.data.preprocessor import CVMPreprocessor
from src.cvm.models.churn_predictor import ChurnPredictor
from src.cvm.models.segmentation import CustomerSegmentation
from src.cvm.models.propensity import PropensityModel
from src.cvm.models.clv_predictor import CLVPredictor

logger = logging.getLogger(__name__)

class CVMModelRegistry:
    def __init__(self):
        self.preprocessor = None
        self.churn_model = None
        self.segmentation_model = None
        self.propensity_bundle = None
        self.propensity_topup = None
        self.propensity_reactivation = None
        self.clv_model = None
        self.master_df = None
        self.is_loaded = False

registry = CVMModelRegistry()

def load_all_models() -> None:
    """Load all serialised models and feature store at startup."""
    start_time = time.time()
    logger.info("Loading all CVM model files...")
    
    try:
        # 1. Load Preprocessor
        preprocessor = CVMPreprocessor()
        preprocessor = CVMPreprocessor.load(settings.PREPROCESSOR_FILE)
        registry.preprocessor = preprocessor
        
        # 2. Load Churn Predictor
        churn_model = ChurnPredictor()
        churn_model.load(settings.CHURN_MODEL_FILE)
        registry.churn_model = churn_model
        
        # 3. Load Customer Segmentation
        seg_model = CustomerSegmentation()
        seg_model.load(settings.SEGMENTATION_MODEL_FILE)
        registry.segmentation_model = seg_model
        
        # 4. Load Propensity Models
        prop_bundle = PropensityModel("bundle_upgrade")
        prop_bundle.load(settings.PROPENSITY_BUNDLE_FILE)
        registry.propensity_bundle = prop_bundle
        
        prop_topup = PropensityModel("voice_topup")
        prop_topup.load(settings.PROPENSITY_TOPUP_FILE)
        registry.propensity_topup = prop_topup
        
        prop_react = PropensityModel("reactivation")
        prop_react.load(settings.PROPENSITY_REACTIVATION_FILE)
        registry.propensity_reactivation = prop_react
        
        # 5. Load CLV Predictor
        clv_model = CLVPredictor()
        clv_model.load(settings.CLV_MODEL_FILE)
        registry.clv_model = clv_model
        
        # 6. Load Master Dataframe
        if settings.FEATURES_FILE.exists():
            registry.master_df = pd.read_csv(settings.FEATURES_FILE)
            logger.info(f"Loaded master feature store from {settings.FEATURES_FILE} containing {len(registry.master_df)} records.")
        else:
            logger.warning(f"Master feature store {settings.FEATURES_FILE} not found. Analytics endpoints will fail.")
            registry.master_df = pd.DataFrame()
            
        registry.is_loaded = True
        duration = time.time() - start_time
        logger.info(f"All models successfully loaded in {duration:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error loading models during startup registry build: {e}", exc_info=True)
        raise RuntimeError(f"Model initialization failure: {e}")

def get_registry() -> CVMModelRegistry:
    """Dependency injector utility for FastAPI endpoints."""
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models registry has not finished initializing yet.")
    return registry
