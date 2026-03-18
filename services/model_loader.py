import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
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
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize and load models"""
        if not self._initialized:
            print("=" * 50)
            print("Loading models...")
            print("=" * 50)
            self._load_models()
            self._initialized = True
    
    def _load_models(self):
        """Load all trained models"""
        
        # Chemins des modeles
        model_paths = {
            'iso_model': ANOMALY_MODEL_PATH,
            'iso_scaler': ANOMALY_SCALER_PATH,
            'lstm_model': RUL_MODEL_PATH,
            'rul_scaler': RUL_SCALER_PATH,
            'params': ANOMALY_PARAMS_PATH
        }
        
        # Verifier si les fichiers existent
        missing_files = []
        for name, path in model_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print("Attention: Fichiers manquants:")
            for f in missing_files:
                print(f"   {f}")
            
            # En CI/CD, on ne peut pas charger les modeles
            if os.environ.get('CI') == 'true':
                print("Mode CI/CD detecte - utilisation de modeles simules")
                self._create_mock_models()
                return
            else:
                raise FileNotFoundError(f"Modeles manquants: {missing_files}")
        
        # Load anomaly detection models
        print(f"Loading Isolation Forest from: {ANOMALY_MODEL_PATH}")
        self.iso_model = joblib.load(ANOMALY_MODEL_PATH)
        print("OK - Isolation Forest loaded")
        
        print(f"Loading ISO Scaler from: {ANOMALY_SCALER_PATH}")
        self.iso_scaler = joblib.load(ANOMALY_SCALER_PATH)
        print("OK - ISO Scaler loaded")
        
        # Load model parameters
        try:
            self.model_params = joblib.load(ANOMALY_PARAMS_PATH)
            print("OK - Model parameters loaded")
        except:
            print("Using default model parameters")
            self.model_params = {
                'p90_normal': -0.0075,
                'p97_normal': 0.0051,
                'THRESHOLD': 0.0051
            }
        
        # Load RUL model
        print(f"Loading LSTM from: {RUL_MODEL_PATH}")
        self.lstm_model = load_model(
            RUL_MODEL_PATH,
            custom_objects={'asymmetric_mse': asymmetric_mse},
            compile=False
        )
        print("OK - LSTM model loaded")
        
        print(f"Loading RUL Scaler from: {RUL_SCALER_PATH}")
        self.rul_scaler = joblib.load(RUL_SCALER_PATH)
        print("OK - RUL Scaler loaded")
        
        print("=" * 50)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 50)
    
    def _create_mock_models(self):
        """Create mock models for CI/CD testing"""
        print("Creation de modeles simules pour les tests CI/CD")
        
        # Mock Isolation Forest
        from sklearn.ensemble import IsolationForest
        import numpy as np
        
        # Creer un mock du modele Isolation Forest
        self.iso_model = IsolationForest(contamination=0.1, random_state=42)
        # Entrainer sur des donnees factices
        X_dummy = np.random.randn(100, 12)
        self.iso_model.fit(X_dummy)
        
        # Mock scaler
        from sklearn.preprocessing import StandardScaler
        self.iso_scaler = StandardScaler()
        self.iso_scaler.fit(X_dummy)
        
        # Mock RUL scaler
        from sklearn.preprocessing import MinMaxScaler
        self.rul_scaler = MinMaxScaler()
        self.rul_scaler.fit(X_dummy)
        
        # Mock LSTM model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model = Sequential([
            LSTM(50, input_shape=(30, 12)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        
        # Mock params
        self.model_params = {
            'p90_normal': -0.0075,
            'p97_normal': 0.0051,
            'THRESHOLD': 0.0051
        }
        
        print("OK - Modeles simules crees avec succes")
    
    def get_anomaly_score(self, features):
        """Get anomaly score from Isolation Forest"""
        if hasattr(self, 'iso_model'):
            score = -self.iso_model.decision_function(features)[0]
            return float(score)
        return 0.0  # Valeur par defaut pour les tests
    
    def predict_rul(self, sequence):
        """Predict RUL from sensor sequence"""
        if hasattr(self, 'lstm_model'):
            pred = self.lstm_model.predict(sequence, verbose=0)[0][0]
            return float(max(0, pred))
        return 100.0  # Valeur par defaut pour les tests

# Fonction pour recuperer l'instance
def get_models():
    """Get or create ModelLoader instance."""
    return ModelLoader()

# Singleton instance
models = get_models()
