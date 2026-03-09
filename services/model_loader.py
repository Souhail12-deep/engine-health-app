import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from config import (
    ANOMALY_MODEL_PATH, ANOMALY_SCALER_PATH, ANOMALY_PARAMS_PATH,
    RUL_MODEL_PATH, RUL_SCALER_PATH
)

# Custom loss function for LSTM
def asymmetric_mse(y_true, y_pred):
    """Penalize late predictions more heavily"""
    error = y_pred - y_true
    late_penalty = 2.0
    loss = tf.where(error > 0, late_penalty * tf.square(error), tf.square(error))
    return tf.reduce_mean(loss)

class ModelLoader:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, skip_load=False):
        """Initialize with option to skip loading for tests."""
        if not self._initialized and not skip_load:
            self._load_models()
            self._initialized = True
    
    def _load_models(self):
        """Load all trained models"""
        print("=" * 50)
        print("Loading models...")
        print("=" * 50)
        
        # Check if files exist before loading
        if not os.path.exists(ANOMALY_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {ANOMALY_MODEL_PATH}")
        
        # Load anomaly detection models
        self.iso_model = joblib.load(ANOMALY_MODEL_PATH)
        self.iso_scaler = joblib.load(ANOMALY_SCALER_PATH)
        
        # Load model parameters
        try:
            self.model_params = joblib.load(ANOMALY_PARAMS_PATH)
        except:
            self.model_params = {
                'p90_normal': -0.0075,
                'p97_normal': 0.0051,
                'THRESHOLD': 0.0051
            }
        
        # Load RUL model
        self.lstm_model = load_model(
            RUL_MODEL_PATH,
            custom_objects={'asymmetric_mse': asymmetric_mse},
            compile=False
        )
        self.rul_scaler = joblib.load(RUL_SCALER_PATH)
        
        print("✅ All models loaded successfully")
    
    def get_anomaly_score(self, features):
        """Get anomaly score from Isolation Forest"""
        score = -self.iso_model.decision_function(features)[0]
        return float(score)
    
    def predict_rul(self, sequence):
        """Predict RUL from sensor sequence"""
        pred = self.lstm_model.predict(sequence, verbose=0)[0][0]
        return float(max(0, pred))

# Don't create instance at import time - will be created when needed
models = None

def get_models():
    """Get or create ModelLoader instance."""
    global models
    if models is None:
        models = ModelLoader()
    return models