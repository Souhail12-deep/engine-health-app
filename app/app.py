#!/usr/bin/env python3
"""
Production-ready Flask application for Engine Health Monitoring (GRU version)
"""
import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Déterminer le chemin absolu du répertoire racine du projet
# On remonte d'un niveau car ce fichier est dans app/app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialiser l'application Flask avec les bons dossiers
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
metrics = PrometheusMetrics(app)

# Register blueprints
from routes.ui import ui_bp
from routes.predict import predict_bp
app.register_blueprint(ui_bp)
app.register_blueprint(predict_bp)

# Import the model loader singleton (this loads all models once)
from services.model_loader import models

# ---------------------- Routes ----------------------
@app.route('/health', methods=['GET'])
def health():
    status = {
        'status': 'healthy',
        'models_loaded': all([
            hasattr(models, 'iso_model') and models.iso_model is not None,
            hasattr(models, 'iso_scaler') and models.iso_scaler is not None,
            hasattr(models, 'lstm_model') and models.lstm_model is not None,
            hasattr(models, 'rul_scaler') and models.rul_scaler is not None,
            hasattr(models, 'scenario_samples') and models.scenario_samples is not None
        ])
    }
    return jsonify(status), 200

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    return metrics.export()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_scenario_sensors', methods=['POST'])
@metrics.counter('scenario_requests', 'Number of scenario requests')
def get_scenario_sensors():
    try:
        data = request.json
        scenario = data.get('scenario', 'normal')
        if not hasattr(models, 'scenario_samples') or models.scenario_samples is None:
            return jsonify({'error': 'Scenario samples not loaded'}), 503
        scenario_data = [s for s in models.scenario_samples if s['scenario'] == scenario]
        if not scenario_data:
            return jsonify({'error': f'No {scenario} samples found'}), 404
        import random
        sample = random.choice(scenario_data)
        return jsonify({
            'engine_id': sample['engine_id'],
            'cycle': sample['cycle'],
            'sensors': sample['sensor_window'][-1]
        })
    except Exception as e:
        logger.error(f"Error in get_scenario_sensors: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyse', methods=['POST'])
@metrics.counter('analysis_requests', 'Number of analysis requests')
def analyse():
    try:
        data = request.json
        engine_id = data.get('engine_id')
        cycle = data.get('cycle')
        if not engine_id or not cycle:
            return jsonify({'error': 'Missing engine_id or cycle'}), 400
        
        if not hasattr(models, 'scenario_samples') or models.scenario_samples is None:
            return jsonify({'error': 'Models not loaded properly'}), 503
        
        sample = None
        for s in models.scenario_samples:
            if s['engine_id'] == engine_id and s['cycle'] == cycle:
                sample = s
                break
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404
        
        sensor_window = sample['sensor_window']
        window_df = pd.DataFrame(sensor_window, columns=[
            's2', 's3', 's4', 's7', 's8', 's9', 
            's11', 's12', 's15', 's17', 's20', 's21'
        ])
        
        from services.preprocessing_service import PreprocessingService
        anomaly_features = PreprocessingService.prepare_features_for_anomaly(window_df, models.iso_scaler)
        anomaly_score = -models.iso_model.decision_function(anomaly_features)[0]
        
        rul_sequence = PreprocessingService.prepare_sequence_for_rul(window_df, models.rul_scaler)
        rul_prediction = float(models.lstm_model.predict(rul_sequence, verbose=0)[0][0])
        
        rul_thresholds = {'WARNING': 80, 'CRITICAL': 30}
        anomaly_thresholds = {'WARNING': -0.0075, 'CRITICAL': 0.0051}
        
        if rul_prediction <= rul_thresholds['CRITICAL']:
            rul_status = 'CRITICAL'
        elif rul_prediction <= rul_thresholds['WARNING']:
            rul_status = 'WARNING'
        else:
            rul_status = 'NORMAL'
        
        if anomaly_score > anomaly_thresholds['CRITICAL']:
            anomaly_status = 'CRITICAL'
        elif anomaly_score > anomaly_thresholds['WARNING']:
            anomaly_status = 'WARNING'
        else:
            anomaly_status = 'NORMAL'
        
        status_priority = {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2}
        final_status = max([rul_status, anomaly_status], key=lambda x: status_priority[x])
        
        from utils.sensor_contribution import calculate_sensor_contributions
        top_sensors, group_severity = calculate_sensor_contributions(window_df.values, anomaly_score)
        
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
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)