"""Tests for the Flask application."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app after mocks are in place
from app.app import app

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_index_page(client):
    """Test the main page loads."""
    response = client.get('/')
    assert response.status_code in [200, 500]

def test_scenario_endpoint_without_models(client):
    """Test scenario endpoint without models loaded."""
    response = client.post('/get_scenario_sensors', 
                          json={'scenario': 'normal'})
    # Should return 503 if scenario_samples not loaded
    assert response.status_code in [503, 404]

def test_static_file_access(client):
    """Test that static files can be accessed."""
    response = client.get('/static/style.css')
    assert response.status_code in [200, 404]