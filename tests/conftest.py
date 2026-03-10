"""Pytest configuration and fixtures."""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_all_models():
    """Mock all model loading for tests."""
    mock_model = MagicMock()
    mock_model.decision_function.return_value = [0.5]
    mock_model.predict.return_value = [[100]]
    
    with patch('services.model_loader.joblib.load', return_value=mock_model), \
         patch('services.model_loader.tf.keras.models.load_model', return_value=mock_model), \
         patch('os.path.exists', return_value=True):
        yield

@pytest.fixture
def app():
    """Create test app with test templates."""
    from app.app import app
    
    # Set test template folder
    test_template_dir = os.path.join(os.path.dirname(__file__), 'test_templates')
    app.config['TESTING'] = True
    app.template_folder = test_template_dir
    
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    with app.test_client() as client:
        yield client