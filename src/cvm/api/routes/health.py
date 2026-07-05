from datetime import datetime
import logging
from fastapi import APIRouter, Depends

from src.cvm.api.dependencies import get_registry, CVMModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check(registry: CVMModelRegistry = Depends(get_registry)):
    """Health check endpoint indicating model loaded state, version, and server time."""
    return {
        "status": "healthy",
        "model_loaded_state": registry.is_loaded,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
