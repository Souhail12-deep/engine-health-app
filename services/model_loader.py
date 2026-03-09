import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Custom loss function for LSTM
def asymmetric_mse(y_true, y_pred):
    error = y_pred - y_true
    late_penalty = 2.0
    loss = tf.where(error > 0, late_penalty * tf.square(error), tf.square(error))
    return tf.reduce_mean(loss)

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Load all trained models"""
        print("=" * 50)
        print("Loading models in model_loader.py...")
        print("=" * 50)
        
        base_path = '/app/models'
        
        # Define paths
        self.anomaly_model_path = os.path.join(base_path, 'anodet_models', 'isolation_forest_latest.pkl')
        self.anomaly_scaler_path = os.path.join(base_path, 'anodet_models', 'scaler_latest.pkl')
        self.rul_model_path = os.path.join(base_path, 'rul_models', 'lstm_rul_latest.keras')
        self.rul_scaler_path = os.path.join(base_path, 'rul_models', 'minmax_scaler_latest.pkl')
        
        print(f"Loading Isolation Forest from: {self.anomaly_model_path}")
        self.iso_model = joblib.load(self.anomaly_model_path)
        print("✅ Isolation Forest loaded")
        
        print(f"Loading ISO Scaler from: {self.anomaly_scaler_path}")
        self.iso_scaler = joblib.load(self.anomaly_scaler_path)
        print("✅ ISO Scaler loaded")
        
        print(f"Loading LSTM from: {self.rul_model_path}")
        self.lstm_model = load_model(
            self.rul_model_path,
            custom_objects={'asymmetric_mse': asymmetric_mse},
            compile=False
        )
        print("✅ LSTM model loaded")
        
        print(f"Loading RUL Scaler from: {self.rul_scaler_path}")
        self.rul_scaler = joblib.load(self.rul_scaler_path)
        print("✅ RUL Scaler loaded")
        
        print("=" * 50)
        print("✅✅✅ ALL MODELS LOADED SUCCESSFULLY IN MODEL_LOADER! ✅✅✅")
        print("=" * 50)
    
    def get_anomaly_score(self, features):
        score = -self.iso_model.decision_function(features)[0]
        return float(score)
    
    def predict_rul(self, sequence):
        pred = self.lstm_model.predict(sequence, verbose=0)[0][0]
        return float(max(0, pred))

# Singleton instance
print("Creating ModelLoader instance...")
models = ModelLoader()