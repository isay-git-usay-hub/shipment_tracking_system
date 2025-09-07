"""
Basic API Tests for Maersk Shipment AI System
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ml_status():
    """Test ML model status endpoint"""
    response = client.get("/ml/status")
    assert response.status_code == 200
    data = response.json()
    # Check for model availability
    assert "model_available" in data or "model_ready" in data


def test_ml_predict_minimal():
    """Test ML prediction with minimal data"""
    response = client.post("/ml/predict", json={})
    # API accepts empty data and uses defaults
    assert response.status_code == 200


def test_ml_predict_valid():
    """Test ML prediction with valid shipment data"""
    shipment_data = {
        "shipment_id": "TEST001",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "waiting_time": 15,
        "timestamp": "2025-09-07T10:00:00",
        "asset_id": "Truck_1",
        "traffic_conditions": "Moderate",
        "weather_conditions": "Clear"
    }
    
    response = client.post("/ml/predict", json=shipment_data)
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure for batch predictions
    assert "predictions" in data or "delay_probability" in data
    if "predictions" in data:
        assert isinstance(data["predictions"], list)
        if len(data["predictions"]) > 0:
            first_pred = data["predictions"][0]
            assert "delay_probability" in first_pred
            assert 0 <= first_pred["delay_probability"] <= 1


def test_analytics_status():
    """Test analytics system status"""
    response = client.get("/analytics/status")
    assert response.status_code == 200
    data = response.json()
    assert "analytics_available" in data


def test_data_validate():
    """Test data validation endpoint"""
    response = client.get("/data/validate")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data


def test_shipments_list():
    """Test shipments listing endpoint"""
    response = client.get("/shipments/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)




if __name__ == "__main__":
    # Run tests
    print("Running API tests...")
    test_health_check()
    print("✓ Health check passed")
    
    test_ml_status()
    print("✓ ML status check passed")
    
    test_ml_predict_minimal()
    print("✓ Minimal prediction test passed")
    
    test_ml_predict_valid()
    print("✓ Valid prediction test passed")
    
    test_analytics_status()
    print("✓ Analytics status check passed")
    
    test_data_validate()
    print("✓ Data validation check passed")
    
    test_shipments_list()
    print("✓ Shipments list test passed")
    
    print("\n✅ All API tests passed!")
