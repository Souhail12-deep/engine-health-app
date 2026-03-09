import numpy as np
import pandas as pd
import os
import random
import pickle
from config import SELECTED_SENSORS, WINDOW_SIZE
from services.preprocessing_service import PreprocessingService
from services.model_loader import get_models
from utils.sensor_contribution import calculate_sensor_contributions
from utils.severity import determine_status

# Load pre-processed scenario windows
WINDOWS_PATH = os.path.join("data", "test", "scenario_windows.pkl")

if os.path.exists(WINDOWS_PATH):
    with open(WINDOWS_PATH, 'rb') as f:
        scenario_windows = pickle.load(f)
    print(f"✅ Loaded {len(scenario_windows)} pre-processed scenario windows")
    
    # Convert to DataFrame for easier filtering
    scenario_df = pd.DataFrame(scenario_windows)
else:
    scenario_windows = None
    scenario_df = None
    print("⚠️ No pre-processed windows found. Run utils/preprocess_test_data.py first")

def get_sensor_values(scenario_type="normal"):
    """Get sensor values for a specific scenario."""
    if scenario_df is None or len(scenario_df) == 0:
        return None, None, None
    
    scenario_data = scenario_df[scenario_df['scenario'] == scenario_type]
    if len(scenario_data) == 0:
        return None, None, None
    
    sample_idx = random.randint(0, len(scenario_data) - 1)
    sample = scenario_data.iloc[sample_idx]
    
    engine_id = int(sample['engine_id'])
    cycle = int(sample['cycle'])
    sensor_window = sample['sensor_window']
    
    last_row_sensors = sensor_window[-1] if isinstance(sensor_window, list) else sensor_window
    return engine_id, cycle, last_row_sensors

def run_analysis(engine_id, cycle):
    """Run complete analysis for a given engine and cycle."""
    if scenario_df is None:
        return {"error": "No pre-processed windows available"}
    
    matches = scenario_df[(scenario_df['engine_id'] == engine_id) & 
                          (scenario_df['cycle'] == cycle)]
    
    if len(matches) == 0:
        return {"error": f"No pre-processed window found for Engine {engine_id} at cycle {cycle}"}
    
    sample = matches.iloc[0]
    sensor_window = sample['sensor_window']
    window_df = pd.DataFrame(sensor_window, columns=SELECTED_SENSORS)
    
    # Get models (lazy loading)
    models = get_models()
    
    # Prepare features
    anomaly_features = PreprocessingService.prepare_features_for_anomaly(window_df, models.iso_scaler)
    rul_sequence = PreprocessingService.prepare_sequence_for_rul(window_df, models.rul_scaler)
    
    # Get predictions
    anomaly_score = models.get_anomaly_score(anomaly_features)
    rul_prediction = models.predict_rul(rul_sequence)
    
    # Determine status
    final_status, rul_status, anomaly_status = determine_status(rul_prediction, anomaly_score)
    
    # Get top contributing sensors
    raw_matrix = window_df[SELECTED_SENSORS].values
    top_sensors, group_severity = calculate_sensor_contributions(raw_matrix, anomaly_score)
    
    # Get last row sensor values
    last_row = window_df.iloc[-1]
    sensor_values = {s: float(last_row[s]) for s in SELECTED_SENSORS}
    
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "status": final_status,
        "rul_status": rul_status,
        "anomaly_status": anomaly_status,
        "rul": round(rul_prediction, 1),
        "anomaly_score": round(anomaly_score, 4),
        "top_sensors": top_sensors,
        "group_severity": group_severity,
        "sensors": sensor_values
    }