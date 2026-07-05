import json
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from src.cvm.api.dependencies import get_registry, CVMModelRegistry
from src.cvm.campaign.opportunity_base import OpportunityBaseGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/campaigns", tags=["Campaigns"])

@router.get("/opportunity-base")
def get_opportunity_base(
    campaign_type: str = Query(..., description="One of: churn_retention, bundle_upsell, voice_topup, reactivation, loyalty_reward"),
    max_size: int = Query(1000, ge=1, le=10000),
    registry: CVMModelRegistry = Depends(get_registry)
):
    """Generates a filtered, prioritized, campaign-ready customer list. Returns summary headers."""
    df = registry.master_df
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail="Feature store database is empty.")
        
    try:
        generator = OpportunityBaseGenerator(df)
        opp_df = generator.generate(campaign_type, max_size)
        
        # Calculate summary metrics
        total_customers = len(opp_df)
        avg_propensity = float(opp_df["propensity_score"].mean()) if total_customers > 0 else 0.0
        
        if total_customers > 0:
            channel_breakdown = opp_df["recommended_channel"].value_counts().to_dict()
        else:
            channel_breakdown = {}
            
        headers = {
            "X-Total-Customers": str(total_customers),
            "X-Avg-Propensity-Score": f"{avg_propensity:.4f}",
            "X-Channel-Breakdown": json.dumps(channel_breakdown)
        }
        
        return JSONResponse(
            content=opp_df.to_dict(orient="records"),
            headers=headers
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Error generating campaign opportunity base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error generating campaign base.")

@router.get("/summary", response_model=List[Dict[str, Any]])
def get_campaigns_summary(registry: CVMModelRegistry = Depends(get_registry)):
    """Return campaign eligibility summary indicating targeted customer counts and average propensities."""
    df = registry.master_df
    if df is None or len(df) == 0:
        return []
        
    try:
        generator = OpportunityBaseGenerator(df)
        campaign_types = ["churn_retention", "bundle_upsell", "voice_topup", "reactivation", "loyalty_reward"]
        
        summary = []
        for ctype in campaign_types:
            # Generate opportunity base without size cap to get total eligible
            opp_df = generator.generate(ctype, max_size=999999)
            eligible = len(opp_df)
            avg_score = float(opp_df["propensity_score"].mean()) if eligible > 0 else 0.0
            summary.append({
                "campaign": ctype,
                "eligible_customers": eligible,
                "avg_propensity_score": avg_score
            })
            
        return summary
    except Exception as e:
        logger.error(f"Error computing campaign summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving campaigns summary.")

@router.get("/segments/profiles", response_model=List[Dict[str, Any]])
def get_segment_profiles(registry: CVMModelRegistry = Depends(get_registry)):
    """Return K-Means cluster profile summaries and mapped campaign recommendations."""
    seg_model = registry.segmentation_model
    if not seg_model.is_fitted or seg_model.cluster_profiles.empty:
        raise HTTPException(status_code=404, detail="Customer segmentation model has not been trained or profiled yet.")
        
    try:
        profiles_df = seg_model.cluster_profiles
        profiles_dict = profiles_df.to_dict(orient="index")
        
        result = []
        for cluster_id, metrics in profiles_dict.items():
            camp_info = seg_model._campaign_mapping.get(int(cluster_id), {})
            metrics_copy = metrics.copy()
            metrics_copy["cluster_id"] = int(cluster_id)
            metrics_copy["segment_label"] = camp_info.get("label", f"Cluster {cluster_id}")
            metrics_copy["description"] = camp_info.get("description", "Customer behavioral segment")
            metrics_copy["recommended_campaign"] = camp_info.get("recommended_campaign", "Standard campaign")
            result.append(metrics_copy)
            
        return result
    except Exception as e:
        logger.error(f"Error building segment profiles details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving segment profiles.")
