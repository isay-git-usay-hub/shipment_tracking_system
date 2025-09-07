"""
Main test file for testing the API
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from api.main import app
from models.database.connection import get_db
from models.schemas import CustomerCreate, ShipmentCreate, PredictionRequest

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

    def test_liveness_probe(self):
        """Test liveness probe"""
        response = client.get("/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestCustomerEndpoints:
    """Test customer management endpoints"""

    def test_create_customer(self):
        """Test customer creation"""
        customer_data = {
            "name": "Test Customer",
            "email": "test@example.com",
            "company": "Test Company"
        }

        response = client.post("/api/customers/", json=customer_data)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == customer_data["name"]
        assert data["email"] == customer_data["email"]

    def test_get_customers(self):
        """Test getting customers list"""
        response = client.get("/api/customers/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_customer_by_id(self):
        """Test getting customer by ID"""
        # First create a customer
        customer_data = {
            "name": "Test Customer 2",
            "email": "test2@example.com"
        }

        create_response = client.post("/api/customers/", json=customer_data)
        assert create_response.status_code == 201
        customer_id = create_response.json()["id"]

        # Then get it
        response = client.get(f"/api/customers/{customer_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == customer_id


class TestShipmentEndpoints:
    """Test shipment management endpoints"""

    def test_get_shipments(self):
        """Test getting shipments list"""
        response = client.get("/api/shipments/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_shipment_without_customer(self):
        """Test creating shipment without existing customer fails"""
        shipment_data = {
            "shipment_id": "TEST001",
            "customer_id": 99999,  # Non-existent customer
            "origin_port": "Shanghai",
            "destination_port": "Rotterdam",
            "scheduled_departure": datetime.now().isoformat(),
            "scheduled_arrival": datetime.now().isoformat()
        }

        response = client.post("/api/shipments/", json=shipment_data)
        assert response.status_code == 400


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    def test_predict_delay(self):
        """Test delay prediction"""
        prediction_request = {
            "origin_port": "Shanghai",
            "destination_port": "Rotterdam", 
            "scheduled_departure": datetime.now().isoformat(),
            "scheduled_arrival": datetime.now().isoformat(),
            "cargo_weight": 1000
        }

        response = client.post("/api/predictions/delay", json=prediction_request)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_delay_hours" in data
        assert "delay_probability" in data
        assert "confidence_score" in data

    def test_batch_predict(self):
        """Test batch prediction"""
        requests = [
            {
                "origin_port": "Shanghai",
                "destination_port": "Rotterdam", 
                "scheduled_departure": datetime.now().isoformat(),
                "scheduled_arrival": datetime.now().isoformat()
            },
            {
                "origin_port": "Singapore",
                "destination_port": "Hamburg", 
                "scheduled_departure": datetime.now().isoformat(),
                "scheduled_arrival": datetime.now().isoformat()
            }
        ]

        response = client.post("/api/predictions/batch", json=requests)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


class TestDashboardEndpoints:
    """Test dashboard endpoints"""

    def test_dashboard_stats(self):
        """Test dashboard statistics"""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_shipments" in data
        assert "active_shipments" in data
        assert "total_customers" in data

    def test_recent_shipments(self):
        """Test recent shipments"""
        response = client.get("/api/dashboard/shipments/recent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_dashboard_alerts(self):
        """Test dashboard alerts"""
        response = client.get("/api/dashboard/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total_alerts" in data


if __name__ == "__main__":
    pytest.main([__file__])
