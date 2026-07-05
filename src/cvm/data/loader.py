import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Custom Exception for data loading errors"""    
    pass
    

class DataValidationError(Exception):
    """Custom Exception for data validation errors"""
    pass


IBM_REQUIRED_COLUMNS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
       "tenure", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
       "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
       "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]

EVENTS_REQUIRED_COLUMNS = ["customer_id", "event_date", "event_type", "recharge_amount",
                "data_mb_used", "call_duration_minutes", "channel",
                "offer_presented", "offer_accepted"]

class CVMDataLoader:
    """Handles loading of raw data files into pandas DataFrames"""

    def load_ibm_churn(self, path: Path) -> pd.DataFrame:
        """Load and validate IBM Telco Churn Dataset"""
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        try:
            df = pd.read_csv(path)
            self._validate_ibm_churn(df)     
            return df
        except Exception as e:
            raise DataLoadError(f"Error loading IBM churn data: {e}")
    
    def _validate_ibm_churn(self, df: pd.DataFrame) -> None:
        """Validate IBM churn dataset structure and basic quality"""
        missing_cols = set(IBM_REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
        df["customerID"] = df["customerID"].str.strip().astype(object) 
        df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int) 
        churn_distribution = df["Churn"].value_counts()
        logger.info(f"Churn distribution:\n{churn_distribution}")

        return df

    def load_events(self, path: Path) -> pd.DataFrame:
        """Load and validate customer events data"""
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        try:
            df = pd.read_csv(path)
            self._validate_events(df)
            return df
        except Exception as e:
            raise DataLoadError(f"Error loading events data: {e}")
    
    def _validate_events(self, df: pd.DataFrame) -> None:
        """Validate customer events data structure and basic quality"""
        missing_cols = set(EVENTS_REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df["recharge_amount"] = pd.to_numeric(df["recharge_amount"], errors="coerce")
        df["data_mb_used"] = pd.to_numeric(df["data_mb_used"], errors="coerce")
        df["call_duration_minutes"] = pd.to_numeric(df["call_duration_minutes"], errors="coerce")
        df["offer_presented"] = df["offer_presented"].astype(object)  # Ensure non-numeric
        df["offer_accepted"] = df["offer_accepted"].astype(object)

        # Validate event type distribution
        event_type_counts = df["event_type"].value_counts()
        logger.info(f"\nEvent type distribution:\n{event_type_counts}")

        return df
    
    def load_both(self, ibm_path: Path, events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load both IBM churn data and events data"""
        ibm_df = self.load_ibm_churn(ibm_path)
        events_df = self.load_events(events_path)
        # Verify that customerID columns match
        if not ibm_df["customerID"].equals(events_df["customer_id"]):
            raise DataValidationError("Customer IDs do not match between datasets")
        return ibm_df, events_df
    
    
# Wrapper functions
def load_ibm(path: Path) -> pd.DataFrame:
    return CVMDataLoader().load_ibm_churn(path)

def load_events(path: Path) -> pd.DataFrame:
    return CVMDataLoader().load_events(path)

def load_both(ibm_path: Path, events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return CVMDataLoader().load_both(ibm_path, events_path)
