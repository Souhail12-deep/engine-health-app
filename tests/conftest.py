"""Pytest configuration and fixtures."""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_all_models():
    """Mock all model loading for tests."""
    mock_model = MagicMock()
    mock_model.decision_function.return_value = [0.5]
    mock_model.predict.return_value = [[100]]
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.1] * 12]
    
    with patch('services.model_loader.joblib.load', return_value=mock_model), \
         patch('services.model_loader.tf.keras.models.load_model', return_value=mock_model), \
         patch('os.path.exists', return_value=True):
        yield

@pytest.fixture
def client():
    """Create a test client with test templates."""
    from app.app import app
    app.config['TESTING'] = True
    
    # Use test templates
    test_template_dir = os.path.join(os.path.dirname(__file__), 'test_templates')
    if os.path.exists(test_template_dir):
        app.template_folder = test_template_dir
    
    with app.test_client() as client:
        yield client