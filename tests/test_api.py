import pytest
from fastapi.testclient import TestClient

from src.cvm.api.main import app
from src.cvm.api.dependencies import get_registry

@pytest.fixture
def client(mock_registry):
    """Override get_registry dependency to inject the mock registry for testing routes."""
    app.dependency_overrides[get_registry] = lambda: mock_registry
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

def test_health_returns_200(client):
    """Verify health endpoint check status resolves successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded_state"] is True

def test_customer_profile_returns_200_for_known_id(client):
    """Verify profile retrieval for an existing customer resolves successfully."""
    response = client.get("/api/customers/cust_0001/profile")
    assert response.status_code == 200
    assert response.json()["customer_id"] == "cust_0001"
    assert response.json()["risk_tier"] == "High"  # score 0.75 maps to High (>0.65)
    assert response.json()["clv_tier"] == "Platinum"

def test_customer_profile_returns_404_for_unknown_id(client):
    """Verify profile lookup for a non-existent customer results in a 404."""
    response = client.get("/api/customers/non_existent_id/profile")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_at_risk_customers_all_above_threshold(client):
    """Verify that at-risk customer filters return only targets exceeding the score threshold."""
    response = client.get("/api/customers/at-risk?min_churn_score=0.50&limit=5")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1
    assert records[0]["customer_id"] == "cust_0001"
    assert records[0]["churn_risk_score"] == 0.75

def test_campaign_opportunity_base_valid_campaign_type(client):
    """Verify that generating opportunity bases for a valid campaign returns 200 with headers."""
    response = client.get("/api/campaigns/opportunity-base?campaign_type=churn_retention&max_size=5")
    assert response.status_code == 200
    assert "X-Total-Customers" in response.headers
    assert "X-Avg-Propensity-Score" in response.headers
    assert len(response.json()) > 0

def test_campaign_opportunity_base_invalid_type_returns_422(client):
    """Verify that requesting an invalid campaign type yields a validation error."""
    response = client.get("/api/campaigns/opportunity-base?campaign_type=invalid_campaign")
    assert response.status_code == 422

def test_model_performance_endpoint_returns_dict(client):
    """Verify model performance metrics endpoint returns a valid populated dictionary."""
    response = client.get("/api/models/performance")
    assert response.status_code == 200
    data = response.json()
    assert "churn_predictor" in data
    assert "clv_predictor" in data
    assert data["churn_predictor"]["model_type"] == "XGBClassifier"

def test_segments_profiles_endpoint(client):
    """Verify segment profiles endpoint lists the clustered profiling categories."""
    response = client.get("/api/campaigns/segments/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert len(profiles) > 0
    assert profiles[0]["segment_label"] == "High Spend Data Users"
