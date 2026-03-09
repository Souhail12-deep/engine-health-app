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
    # Mock joblib.load
    mock_joblib = MagicMock()
    mock_model = MagicMock()
    mock_model.decision_function.return_value = [0.5]
    mock_model.predict.return_value = [[100]]
    mock_joblib.return_value = mock_model
    
    # Mock keras load_model
    mock_keras = MagicMock(return_value=mock_model)
    
    # Mock os.path.exists to return True for model paths
    mock_exists = MagicMock(return_value=True)
    
    with patch('services.model_loader.joblib.load', mock_joblib), \
         patch('services.model_loader.load_model', mock_keras), \
         patch('os.path.exists', mock_exists):
        
        # Now import modules that might load models
        from services.model_loader import ModelLoader
        ModelLoader._initialized = False  # Reset for tests
        
        yield