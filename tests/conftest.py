"""Pytest configuration and fixtures."""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test mode environment variable
os.environ['PYTEST_CURRENT_TEST'] = 'true'

@pytest.fixture(autouse=True)
def mock_models():
    """Mock only the model loading - not Flask internals."""
    mock_model = MagicMock()
    mock_model.decision_function.return_value = [0.5]
    mock_model.predict.return_value = [[100]]
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.1] * 12]
    
    # Only mock our specific modules, not builtins or Flask
    with patch('services.model_loader.joblib.load', return_value=mock_model), \
         patch('services.model_loader.tf.keras.models.load_model', return_value=mock_model), \
         patch('services.inference_service.pd.read_csv'), \
         patch('services.inference_service.pickle.load', return_value=[]), \
         patch('os.path.exists', return_value=True):
        yield

@pytest.fixture
def app():
    """Create test app with test templates."""
    # Import app after mocks are in place
    from app.app import app
    
    # Set test template folder
    test_template_dir = os.path.join(os.path.dirname(__file__), 'test_templates')
    os.makedirs(test_template_dir, exist_ok=True)
    
    # Create test template if it doesn't exist
    template_path = os.path.join(test_template_dir, 'index.html')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head><title>Engine Health Test</title></head>
<body>
    <h1>Engine Health Test</h1>
    <p>This is a test template for CI/CD</p>
</body>
</html>''')
    
    app.config['TESTING'] = True
    app.template_folder = test_template_dir
    
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    with app.test_client() as client:
        yield client