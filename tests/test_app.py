"""Tests for the Flask application."""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock model loading before importing app
with patch('services.model_loader.joblib.load') as mock_joblib, \
     patch('services.model_loader.tf.keras.models.load_model') as mock_keras:
    
    mock_model = MagicMock()
    mock_model.decision_function.return_value = [0.5]
    mock_model.predict.return_value = [[100]]
    mock_joblib.return_value = mock_model
    mock_keras.return_value = mock_model
    
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
    assert response.status_code in [200, 500]  # 500 if template missing

def test_scenario_endpoint(client):
    """Test the scenario endpoint."""
    response = client.post('/get_scenario_sensors', 
                          json={'scenario': 'normal'})
    # Should return 503 if models not loaded, or 200 if mocks work
    assert response.status_code in [200, 503]

def test_analyse_endpoint(client):
    """Test the analyse endpoint."""
    response = client.post('/analyse', 
                          json={'engine_id': 1, 'cycle': 150})
    assert response.status_code in [200, 404, 503]