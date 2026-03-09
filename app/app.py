#!/usr/bin/env python3
"""
Production-ready Flask application for Engine Health Monitoring
"""
import os
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='/app/templates',
            static_folder='/app/static')
metrics = PrometheusMetrics(app)

# Register blueprints
from routes.ui import ui_bp
from routes.predict import predict_bp
app.register_blueprint(ui_bp)
app.register_blueprint(predict_bp)

# Global variables for models - lazy loading
iso_model = None
iso_scaler = None
lstm_model = None
rul_scaler = None
scenario_samples = None
MODELS_PATH = '/app/models'

def load_models():
    """Load ML models - called only when needed"""
    global iso_model, iso_scaler, lstm_model, rul_scaler, scenario_samples
    
    logger.info("=" * 50)
    logger.info("STARTING MODEL LOADING PROCESS")
    logger.info("=" * 50)
    
    # Define model paths
    iso_model_path = os.path.join(MODELS_PATH, 'anodet_models', 'isolation_forest_latest.pkl')
    iso_scaler_path = os.path.join(MODELS_PATH, 'anodet_models', 'scaler_latest.pkl')
    lstm_model_path = os.path.join(MODELS_PATH, 'rul_models', 'lstm_rul_latest.keras')
    rul_scaler_path = os.path.join(MODELS_PATH, 'rul_models', 'minmax_scaler_latest.pkl')
    scenario_path = os.path.join(MODELS_PATH, 'scenario_windows.pkl')
    
    # Check if files exist
    logger.info(f"Checking model files:")
    logger.info(f"  Isolation Forest: {os.path.exists(iso_model_path)} - {iso_model_path}")
    logger.info(f"  ISO Scaler: {os.path.exists(iso_scaler_path)} - {iso_scaler_path}")
    logger.info(f"  LSTM Model: {os.path.exists(lstm_model_path)} - {lstm_model_path}")
    logger.info(f"  RUL Scaler: {os.path.exists(rul_scaler_path)} - {rul_scaler_path}")
    logger.info(f"  Scenario Windows: {os.path.exists(scenario_path)} - {scenario_path}")
    
    # Load models
    try:
        # Load Isolation Forest
        logger.info("Loading Isolation Forest model...")
        iso_model = joblib.load(iso_model_path)
        logger.info("✅ Isolation Forest loaded")
        
        # Load ISO Scaler
        logger.info("Loading ISO Scaler...")
        iso_scaler = joblib.load(iso_scaler_path)
        logger.info("✅ ISO Scaler loaded")
        
        # Load LSTM model
        logger.info("Loading LSTM model...")
        lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
        logger.info("✅ LSTM model loaded")
        
        # Load RUL Scaler
        logger.info("Loading RUL Scaler...")
        rul_scaler = joblib.load(rul_scaler_path)
        logger.info("✅ RUL Scaler loaded")
        
        # Load scenario samples
        logger.info("Loading scenario samples...")
        with open(scenario_path, 'rb') as f:
            import pickle
            scenario_samples = pickle.load(f)
        logger.info(f"✅ Loaded {len(scenario_samples)} scenario samples")
        
        logger.info("=" * 50)
        logger.info("✅✅✅ ALL MODELS LOADED SUCCESSFULLY! ✅✅✅")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Lazy loading decorator for routes that need models
def with_models(f):
    def decorated(*args, **kwargs):
        if iso_model is None:
            load_models()
        return f(*args, **kwargs)
    return decorated

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'models_loaded': all([iso_model, iso_scaler, lstm_model, rul_scaler, scenario_samples]) if iso_model else False
    }
    return jsonify(status), 200

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return metrics.export()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/get_scenario_sensors', methods=['POST'])
@metrics.counter('scenario_requests', 'Number of scenario requests')
def get_scenario_sensors():
    """Get sensors for specific scenario"""
    try:
        data = request.json
        scenario = data.get('scenario', 'normal')
        
        if scenario_samples is None:
            return jsonify({'error': 'Scenario samples not loaded'}), 503
        
        # Filter by scenario
        scenario_data = [s for s in scenario_samples if s['scenario'] == scenario]
        
        if not scenario_data:
            return jsonify({'error': f'No {scenario} samples found'}), 404
        
        # Pick random sample
        import random
        sample = random.choice(scenario_data)
        
        return jsonify({
            'engine_id': sample['engine_id'],
            'cycle': sample['cycle'],
            'sensors': sample['sensor_window'][-1]  # Last row of sensors
        })
    
    except Exception as e:
        logger.error(f"Error in get_scenario_sensors: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyse', methods=['POST'])
@metrics.counter('analysis_requests', 'Number of analysis requests')
@with_models
def analyse():
    """Run analysis on engine data"""
    try:
        data = request.json
        engine_id = data.get('engine_id')
        cycle = data.get('cycle')
        
        if not engine_id or not cycle:
            return jsonify({'error': 'Missing engine_id or cycle'}), 400
        
        # Find matching sample
        sample = None
        for s in scenario_samples:
            if s['engine_id'] == engine_id and s['cycle'] == cycle:
                sample = s
                break
        
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404
        
        # Get window data
        sensor_window = sample['sensor_window']
        window_df = pd.DataFrame(sensor_window, columns=[
            's2', 's3', 's4', 's7', 's8', 's9', 
            's11', 's12', 's15', 's17', 's20', 's21'
        ])
        
        # Prepare features for anomaly detection
        from services.preprocessing_service import PreprocessingService
        anomaly_features = PreprocessingService.prepare_features_for_anomaly(
            window_df, iso_scaler
        )
        anomaly_score = -iso_model.decision_function(anomaly_features)[0]
        
        # Prepare for RUL prediction
        rul_sequence = PreprocessingService.prepare_sequence_for_rul(
            window_df, rul_scaler
        )
        rul_prediction = float(lstm_model.predict(rul_sequence, verbose=0)[0][0])
        
        # Determine status
        rul_thresholds = {'WARNING': 80, 'CRITICAL': 30}
        anomaly_thresholds = {'WARNING': -0.0075, 'CRITICAL': 0.0051}
        
        # RUL status
        if rul_prediction <= rul_thresholds['CRITICAL']:
            rul_status = 'CRITICAL'
        elif rul_prediction <= rul_thresholds['WARNING']:
            rul_status = 'WARNING'
        else:
            rul_status = 'NORMAL'
        
        # Anomaly status
        if anomaly_score > anomaly_thresholds['CRITICAL']:
            anomaly_status = 'CRITICAL'
        elif anomaly_score > anomaly_thresholds['WARNING']:
            anomaly_status = 'WARNING'
        else:
            anomaly_status = 'NORMAL'
        
        # Final status (most severe)
        status_priority = {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2}
        final_status = max([rul_status, anomaly_status], 
                          key=lambda x: status_priority[x])
        
        # Calculate top sensors
        from utils.sensor_contribution import calculate_sensor_contributions
        top_sensors, group_severity = calculate_sensor_contributions(
            window_df.values, anomaly_score
        )
        
        # Get last sensor values
        last_row = window_df.iloc[-1]
        sensor_values = {s: float(last_row[s]) for s in window_df.columns}
        
        return jsonify({
            'engine_id': engine_id,
            'cycle': cycle,
            'status': final_status,
            'rul_status': rul_status,
            'anomaly_status': anomaly_status,
            'rul': round(rul_prediction, 1),
            'anomaly_score': round(float(anomaly_score), 4),
            'top_sensors': top_sensors,
            'group_severity': group_severity,
            'sensors': sensor_values
        })
    
    except Exception as e:
        logger.error(f"Error in analyse: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False)