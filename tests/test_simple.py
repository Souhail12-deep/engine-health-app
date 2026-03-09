"""Simple tests that don't require model loading."""

def test_imports():
    """Test that modules can be imported with mocking."""
    try:
        # Mock model loading before imports
        from unittest.mock import patch, MagicMock
        
        with patch('services.model_loader.joblib.load'), \
             patch('services.model_loader.tf.keras.models.load_model'):
            
            import app.app
            import routes.ui
            import routes.predict
            import services.model_loader
            
            print("✅ All imports successful")
            assert True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        assert False

def test_flask_app_config():
    """Test Flask app configuration."""
    from unittest.mock import patch
    
    with patch('services.model_loader.ModelLoader'):
        from app.app import app
        assert app is not None
        assert app.template_folder == '/app/templates'
        assert app.static_folder == '/app/static'
        print("✅ Flask app configured correctly")