import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field

from src.cvm.api.dependencies import get_registry, CVMModelRegistry
from src.cvm.models.churn_predictor import ChurnPrediction
from src.cvm.models.segmentation import SegmentResult
from src.cvm.models.propensity import PropensityResult
from src.cvm.models.clv_predictor import CLVPrediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/customers", tags=["Customers"])

class CustomerProfileResponse(BaseModel):
    customer_id: str
    tenure_months: int
    churn_risk_score: float
    churn_predicted: bool
    risk_tier: str
    clv_90d: float
    clv_tier: str
    rfm_segment: str
    ml_segment: str
    propensity_bundle: float
    propensity_topup: float
    propensity_reactivation: float
    lifecycle_stage: str

class ShapExplanationItem(BaseModel):
    feature: str
    shap_value: float
    direction: str

class BatchScoreRequest(BaseModel):
    features: List[Dict[str, Any]] = Field(
        ...,
        max_items=500,
        description="List of preprocessed features for up to 500 customers",
        json_schema_extra={
            "example": [
                {
                    "customer_id": "7590-VHVEG",
                    "tenure_months": 1,
                    "monthly_charges": 29.85,
                    "total_charges": 29.85,
                    "recharge_count_90d": 3,
                    "total_recharge_90d": 1500.0,
                    "avg_recharge_amount": 500.0,
                    "days_since_last_recharge": 10,
                    "data_usage_mb_90d": 250.0,
                    "data_usage_mb_30d": 100.0,
                    "avg_session_mb": 25.0,
                    "active_days_90d": 5,
                    "app_usage_rate": 0.2,
                    "recharge_trend": -0.05,
                    "r_score": 3,
                    "f_score": 2,
                    "m_score": 2,
                    "rfm_score": 7,
                    "arpu": 500.0,
                    "engagement_score": 0.35,
                    "declining_recharge": 1,
                    "long_silence": 0,
                    "low_activity": 1,
                    "preferred_channel_agent": 0,
                    "preferred_channel_ussd": 1,
                    "preferred_channel_app": 0,
                    "preferred_channel_web": 0,
                    "gender_male": 0,
                    "Partner": 1,
                    "Dependents": 0,
                    "PhoneService": 1,
                    "MultipleLines_No phone service": 0,
                    "MultipleLines_Yes": 0,
                    "internet_dsl": 1,
                    "internet_fiber": 0,
                    "internet_none": 0,
                    "OnlineSecurity": 0,
                    "OnlineBackup": 1,
                    "DeviceProtection": 0,
                    "TechSupport": 0,
                    "StreamingTV": 0,
                    "StreamingMovies": 0,
                    "contract_monthly": 1,
                    "contract_1yr": 0,
                    "contract_2yr": 0,
                    "PaperlessBilling": 1,
                    "payment_method_electronic_check": 1,
                    "payment_method_mailed_check": 0,
                    "payment_method_bank_transfer": 0,
                    "payment_method_credit_card": 0,
                    "offers_presented": 2,
                    "offers_accepted": 1,
                    "offer_acceptance_rate": 0.5,
                    "recharge_count_30d": 1,
                    "total_recharge_30d": 500.0,
                    "days_since_last_activity": 4,
                    "data_usage_trend": 0.1,
                    "app_user": 0,
                    "high_data_user": 0,
                    "premium_customer": 0,
                    "frequent_recharger": 0,
                    "recharge_drop_flag": 1,
                    "high_offer_responder": 0
                }
            ]
        }
    )

class ScoredResult(BaseModel):
    customer_id: str
    churn_risk_score: float
    churn_predicted: bool
    ml_segment: str
    propensity_bundle: float
    propensity_topup: float
    propensity_reactivation: float
    predicted_clv_90d: float
    clv_tier: str
    retention_budget_tier: str

@router.get("/{customer_id}/profile", response_model=CustomerProfileResponse)
def get_customer_profile(
    customer_id: str = Path(
        ...,
        description="The unique subscriber ID key. Examples: 7590-VHVEG, 5575-GNVDE, 3668-QJUXN",
        examples=["7590-VHVEG", "5575-GNVDE", "3668-QJUXN"]
    ),
    registry: CVMModelRegistry = Depends(get_registry)
):
    """Look up a customer profile in the enriched master feature store and return all scores."""
    df = registry.master_df
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="Feature store database is empty.")
        
    customer_row = df[df["customer_id"] == customer_id]
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer with ID '{customer_id}' not found.")
        
    row = customer_row.iloc[0]
    
    # Map churn probability to risk tier
    prob = float(row.get("churn_risk_score", 0.0))
    if prob > 0.8:
        risk_tier = "Critical"
    elif prob > 0.65:
        risk_tier = "High"
    elif prob > 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    return CustomerProfileResponse(
        customer_id=str(row["customer_id"]),
        tenure_months=int(row.get("tenure_months", 0)),
        churn_risk_score=prob,
        churn_predicted=bool(prob >= registry.churn_model.optimal_threshold),
        risk_tier=risk_tier,
        clv_90d=float(row.get("clv_90d", 0.0)),
        clv_tier=str(row.get("clv_tier", "Bronze")),
        rfm_segment=str(row.get("rfm_segment", "Lost")),
        ml_segment=str(row.get("ml_segment", "Dormant")),
        propensity_bundle=float(row.get("propensity_bundle", 0.0)),
        propensity_topup=float(row.get("propensity_topup", 0.0)),
        propensity_reactivation=float(row.get("propensity_reactivation", 0.0)),
        lifecycle_stage=str(row.get("lifecycle_stage", "Active"))
    )

@router.get("/{customer_id}/churn-explanation", response_model=List[ShapExplanationItem])
def get_churn_explanation(
    customer_id: str = Path(
        ...,
        description="The unique subscriber ID key. Examples: 7590-VHVEG, 5575-GNVDE, 3668-QJUXN",
        examples=["7590-VHVEG", "5575-GNVDE", "3668-QJUXN"]
    ),
    registry: CVMModelRegistry = Depends(get_registry)
):
    """Exposes SHAP explanations driving the customer's churn risk score (top 5 features)."""
    df = registry.master_df
    
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="Feature store database is empty.")
        
    customer_rows = df[df["customer_id"] == customer_id]
    if customer_rows.empty:
        raise HTTPException(status_code=404, detail=f"Customer with ID '{customer_id}' not found.")
        
    row_idx = customer_rows.index[0]
    feature_cols = registry.churn_model._feature_names
    
    # Check if SHAP explainer exists
    if registry.churn_model.shap_explainer is None:
        raise HTTPException(status_code=500, detail="SHAP explainer was not fitted on the churn predictor.")
        
    # Extracted feature array for the customer
    x_customer = df.loc[row_idx, feature_cols].values.reshape(1, -1)
    
    # Calculate SHAP values
    shap_vals = registry.churn_model.shap_explainer.shap_values(x_customer)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
        
    shap_vals = np.array(shap_vals).flatten()
    
    # Sort indices by absolute SHAP feature importance
    abs_shap = np.abs(shap_vals)
    top_5_indices = np.argsort(abs_shap)[-5:][::-1]
    
    explanations = []
    for idx in top_5_indices:
        feat_name = feature_cols[idx]
        val = float(shap_vals[idx])
        direction = "increasing" if val > 0 else "decreasing"
        explanations.append(ShapExplanationItem(
            feature=feat_name,
            shap_value=val,
            direction=direction
        ))
        
    return explanations

@router.post("/score", response_model=List[ScoredResult])
def score_batch(payload: BatchScoreRequest, registry: CVMModelRegistry = Depends(get_registry)):
    """Scoring endpoint. Scores a batch of up to 500 customers across all models in real-time."""
    features_batch = payload.features
    if len(features_batch) == 0:
        return []
        
    df_input = pd.DataFrame(features_batch)
    
    feature_cols = registry.churn_model._feature_names
    missing_cols = set(feature_cols) - set(df_input.columns)
    if missing_cols:
        raise HTTPException(status_code=422, detail=f"Missing feature columns for inference: {list(missing_cols)}")
        
    X = df_input[feature_cols].values
    customer_ids = df_input["customer_id"].astype(str).tolist() if "customer_id" in df_input.columns else [str(i) for i in range(len(X))]
    
    # Run predictions on all loaded models
    churn_preds = registry.churn_model.predict(X, customer_ids=customer_ids)
    seg_preds = registry.segmentation_model.predict(X, customer_ids=customer_ids)
    prop_bundle = registry.propensity_bundle.predict(X, customer_ids=customer_ids)
    prop_topup = registry.propensity_topup.predict(X, customer_ids=customer_ids)
    prop_react = registry.propensity_reactivation.predict(X, customer_ids=customer_ids)
    clv_preds = registry.clv_model.predict(X, customer_ids=customer_ids)
    
    scored_results = []
    for i in range(len(X)):
        scored_results.append(ScoredResult(
            customer_id=customer_ids[i],
            churn_risk_score=churn_preds[i].churn_probability,
            churn_predicted=churn_preds[i].churn_predicted,
            ml_segment=seg_preds[i].segment_label,
            propensity_bundle=prop_bundle[i].propensity_score,
            propensity_topup=prop_topup[i].propensity_score,
            propensity_reactivation=prop_react[i].propensity_score,
            predicted_clv_90d=clv_preds[i].predicted_clv_90d,
            clv_tier=clv_preds[i].clv_tier,
            retention_budget_tier=clv_preds[i].retention_budget_tier
        ))
        
    return scored_results

@router.get("/at-risk", response_model=List[Dict[str, Any]])
def get_at_risk_customers(
    min_churn_score: float = Query(0.65, ge=0.0, le=1.0),
    clv_tier: Optional[Literal["Platinum", "Gold", "Silver", "Bronze"]] = Query(None, description="Filter by CLV tier"),
    limit: int = Query(100, ge=1, le=1000),
    registry: CVMModelRegistry = Depends(get_registry)
):
    """Query and return customers exceeding the churn score threshold, sorted by churn risk score descending."""
    df = registry.master_df
    if df is None or len(df) == 0:
        return []
        
    mask = df["churn_risk_score"] >= min_churn_score
    if clv_tier:
        mask = mask & (df["clv_tier"] == clv_tier)
        
    filtered = df[mask].sort_values(by="churn_risk_score", ascending=False).head(limit)
    
    # Return selected descriptive fields
    columns_to_return = [
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "churn_risk_score", "clv_90d", "clv_tier", "rfm_segment", "ml_segment"
    ]
    return filtered[[c for c in columns_to_return if c in filtered.columns]].to_dict(orient="records")
