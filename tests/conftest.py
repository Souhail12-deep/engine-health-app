"""Pytest configuration and fixtures."""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(autouse=True)
def mock_models():
    """Mock model loading for all tests."""
    with patch('services.model_loader.joblib.load') as mock_joblib, \
         patch('services.model_loader.tf.keras.models.load_model') as mock_keras, \
         patch('services.model_loader.ModelLoader._load_models') as mock_load:
        
        # Create mock models
        mock_model = MagicMock()
        mock_model.decision_function.return_value = [0.5]
        mock_model.predict.return_value = [[100]]
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[0.1] * 12]
        
        # Setup return values
        mock_joblib.return_value = mock_model
        mock_keras.return_value = mock_model
        
        # Allow ModelLoader to be created without loading real models
        mock_load.return_value = None
        
        yield