from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BaseCVMModel(ABC):
    def __init__(self):
        self._model = None
        self._feature_names: list = []
        self._trained_at: Optional[datetime] = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the model."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the model."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to path."""
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from path."""
        joblib.load(path)
        logger.info(f"Model loaded from {path}")

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model is fitted."""
        return self._model is not None 

    