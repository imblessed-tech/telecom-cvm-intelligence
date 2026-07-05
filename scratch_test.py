import pandas as pd
import numpy as np
from src.cvm.data.preprocessor import CVMPreprocessor
from src.cvm.features.rfm import RFMAnalyser
from src.cvm.features.behavioural import BehaviouralFeatureEngineer

# Set up logger to see outputs
import logging
logging.basicConfig(level=logging.INFO)

print("1. Creating mock data...")
# Mock IBM Churn dataset
ibm_data = pd.DataFrame({
    'customerID': ['1', '2', '3', '4', '5'],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'SeniorCitizen': [0, 1, 0, 0, 1],
    'Partner': ['Yes', 'No', 'Yes', 'No', 'No'],
    'Dependents': ['No', 'Yes', 'No', 'No', 'No'],
    'tenure': [12, 24, 6, 72, 1],
    'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
    'OnlineSecurity': ['Yes', 'No', 'Yes', 'No', 'No'],
    'OnlineBackup': ['No', 'Yes', 'Yes', 'No', 'No'],
    'DeviceProtection': ['Yes', 'No', 'No', 'No', 'No'],
    'TechSupport': ['Yes', 'Yes', 'No', 'No', 'No'],
    'StreamingTV': ['No', 'No', 'No', 'No', 'No'],
    'StreamingMovies': ['No', 'No', 'No', 'No', 'No'],
    'Contract': ['One year', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month'],
    'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'PaymentMethod': ['Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check'],
    'MonthlyCharges': [50.0, 80.0, 30.0, 20.0, 70.0],
    'TotalCharges': ['600.0', '1920.0', '180.0', '1440.0', '70.0'],
    'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
})
# Make Churn_binary for generate_events or preprocessor
ibm_data['Churn_binary'] = ibm_data['Churn'].map({'Yes': 1, 'No': 0})

# Mock Events dataset
events_data = pd.DataFrame({
    'customer_id': ['1', '1', '2', '2', '3', '3', '4', '4', '5'],
    'event_date': ['2026-07-01', '2026-07-15', '2026-07-05', '2026-07-20', '2026-07-10', '2026-07-22', '2026-07-02', '2026-07-28', '2026-07-03'],
    'event_type': ['RECHARGE', 'DATA_SESSION', 'RECHARGE', 'CALL', 'RECHARGE', 'SMS', 'RECHARGE', 'DATA_SESSION', 'RECHARGE'],
    'recharge_amount': [500.0, None, 1000.0, None, 200.0, None, 1500.0, None, 100.0],
    'data_mb_used': [None, 350.5, None, None, None, None, None, 1200.0, None],
    'call_duration_minutes': [None, None, None, 5.5, None, None, None, None, None],
    'channel': ['APP', 'WEB', 'USSD', 'APP', 'AGENT', 'USSD', 'WEB', 'APP', 'USSD'],
    'offer_presented': [None, 'DATA_BUNDLE_1GB', None, None, None, None, 'VOICE_TOPUP_100', None, None],
    'offer_accepted': [False, True, False, False, False, False, True, False, False]
})

print("2. Running Preprocessor...")
preprocessor = CVMPreprocessor()
cleaned_df = preprocessor.fit_transform(ibm_data, events_data)
print("Preprocessor complete. Cleaned columns:")
print(cleaned_df.columns.tolist())

print("\n3. Running RFM Analyser...")
rfm_analyser = RFMAnalyser()
rfm_df = rfm_analyser.compute(cleaned_df)
print("RFM complete. RFM df shape:", rfm_df.shape)
print("RFM columns:")
print(rfm_df.columns.tolist())
print(rfm_df[['customer_id', 'rfm_score', 'rfm_segment']])

print("\n4. Running Behavioural Feature Engineer...")
# For behavioural engineer, it expects columns in cleaned_df
behavioural_engineer = BehaviouralFeatureEngineer()
behavioural_df = behavioural_engineer.engineer(cleaned_df)
print("Behavioural complete. Behavioural df shape:", behavioural_df.shape)
print("Behavioural columns added:")
print([col for col in behavioural_df.columns if col not in cleaned_df.columns])
print(behavioural_df[['customer_id', 'arpu', 'arpu_tier', 'engagement_score', 'lifecycle_stage']])

print("\nAll tests passed successfully!")
