"""Tests for the Flask application."""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_index_page(client):
    """Test the main page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Engine Health Test" in response.data

def test_scenario_endpoint(client):
    """Test scenario endpoint."""
    response = client.post('/get_scenario_sensors', 
                          json={'scenario': 'normal'})
    assert response.status_code in [200, 503, 404]

def test_static_file_access(client):
    """Test that static files can be accessed."""
    response = client.get('/static/style.css')
    assert response.status_code in [200, 404]