#!/usr/bin/env python3
"""
Terminal test script for Engine Health Monitor
Run this first to verify all components work before launching the Flask app
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("ENGINE HEALTH MONITOR - TERMINAL TEST")
print("="*60)

# ============================================
# 1. TEST DATA LOADING
# ============================================
print("\n[1] TESTING DATA LOADING...")
print("-"*40)

try:
    from utils.test_loader import load_test_data
    
    # Try to load data
    df = load_test_data()
    print(f"✓ Data loaded successfully")
    print(f"  - Shape: {df.shape}")
    print(f"  - Units: {df['unit'].nunique()}")
    print(f"  - Cycles range: {df['cycle'].min()} to {df['cycle'].max()}")
    print(f"  - Columns: {list(df.columns[:10])}...")
    
    DATA_LOADED = True
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    DATA_LOADED = False

# ============================================
# 2. TEST MODEL LOADING
# ============================================
print("\n[2] TESTING MODEL LOADING...")
print("-"*40)

models_loaded = {}

# Load Isolation Forest
try:
    iso_model = joblib.load("models/anodet_models/isolation_forest_latest.pkl")
    iso_scaler = joblib.load("models/anodet_models/scaler_latest.pkl")
    model_params = joblib.load("models/anodet_models/model_params_latest.pkl")
    
    print(f"✓ Isolation Forest loaded")
    print(f"  - Type: {type(iso_model).__name__}")
    print(f"  - Scaler: {type(iso_scaler).__name__}")
    print(f"  - Params: {list(model_params.keys())}")
    
    # Get feature names
    if hasattr(iso_scaler, 'feature_names_in_'):
        iso_features = iso_scaler.feature_names_in_
        print(f"  - Features: {len(iso_features)}")
    else:
        iso_features = None
        print(f"  - Warning: scaler has no feature_names_in_")
    
    models_loaded['iso'] = True
except Exception as e:
    print(f"✗ Failed to load Isolation Forest: {e}")
    models_loaded['iso'] = False

# Load LSTM
try:
    lstm_model = load_model("models/rul_models/lstm_rul_latest.keras", compile=False)
    rul_scaler = joblib.load("models/rul_models/minmax_scaler_latest.pkl")
    
    print(f"✓ LSTM model loaded")
    print(f"  - Input shape: {lstm_model.input_shape}")
    print(f"  - Output shape: {lstm_model.output_shape}")
    print(f"  - Scaler: {type(rul_scaler).__name__}")
    
    if hasattr(rul_scaler, 'feature_names_in_'):
        rul_features = rul_scaler.feature_names_in_
        print(f"  - Features: {len(rul_features)}")
    else:
        rul_features = None
        print(f"  - Warning: scaler has no feature_names_in_")
    
    models_loaded['lstm'] = True
except Exception as e:
    print(f"✗ Failed to load LSTM: {e}")
    models_loaded['lstm'] = False

# ============================================
# 3. TEST FEATURE ENGINEERING
# ============================================
print("\n[3] TESTING FEATURE ENGINEERING...")
print("-"*40)

try:
    from services.feature_engineering import SELECTED_SENSORS, build_engineered_features
    from services.preprocessing_service import prepare_features
    
    print(f"✓ Feature engineering modules imported")
    print(f"  - Selected sensors: {SELECTED_SENSORS}")
    
    # Get a sample window if data is loaded
    if DATA_LOADED:
        # Pick a random engine
        engine_id = df['unit'].iloc[0]
        engine_data = df[df['unit'] == engine_id].sort_values('cycle')
        
        if len(engine_data) >= 30:
            window = engine_data.tail(30).copy()
            
            # Test rename_for_iso
            from services.preprocessing_service import rename_for_iso, add_engineered_features
            
            iso_df = rename_for_iso(window)
            print(f"  ✓ rename_for_iso: {iso_df.shape}")
            
            iso_df_feat = add_engineered_features(iso_df)
            print(f"  ✓ add_engineered_features: {iso_df_feat.shape}")
            
            # Test full pipeline if models are loaded
            if models_loaded['iso'] and models_loaded['lstm']:
                try:
                    iso_scaled, rul_scaled, raw_matrix = prepare_features(
                        window, iso_scaler, rul_scaler
                    )
                    print(f"  ✓ prepare_features successful")
                    print(f"    - ISO input shape: {iso_scaled.shape}")
                    print(f"    - RUL input shape: {rul_scaled.shape}")
                    print(f"    - Raw matrix shape: {raw_matrix.shape}")
                    
                    FEATURES_OK = True
                except Exception as e:
                    print(f"  ✗ prepare_features failed: {e}")
                    FEATURES_OK = False
            else:
                print(f"  ⚠ Skipping prepare_features (models not loaded)")
                FEATURES_OK = False
        else:
            print(f"  ⚠ Engine {engine_id} has only {len(engine_data)} cycles (<30)")
            FEATURES_OK = False
    else:
        print(f"  ⚠ Skipping feature tests (data not loaded)")
        FEATURES_OK = False
        
except Exception as e:
    print(f"✗ Feature engineering test failed: {e}")
    FEATURES_OK = False

# ============================================
# 4. TEST MODEL INFERENCE
# ============================================
print("\n[4] TESTING MODEL INFERENCE...")
print("-"*40)

if DATA_LOADED and FEATURES_OK:
    try:
        # Get anomaly score from Isolation Forest
        anomaly_score = -iso_model.decision_function(iso_scaled)[0]
        print(f"  ✓ Anomaly score: {anomaly_score:.4f}")
        
        # Get RUL prediction from LSTM
        rul_pred = lstm_model.predict(rul_scaled, verbose=0)[0][0]
        print(f"  ✓ RUL prediction: {rul_pred:.2f} cycles")
        
        # Apply thresholds from training
        THRESHOLD = model_params.get('THRESHOLD', 0.01)
        p90_normal = model_params.get('p90_normal', 0.005)
        p97_normal = model_params.get('p97_normal', 0.01)
        
        print(f"\n  Thresholds from training:")
        print(f"    - Alarm threshold: {THRESHOLD:.4f}")
        print(f"    - Warning threshold: {p90_normal:.4f}")
        print(f"    - Critical threshold: {p97_normal:.4f}")
        
        # Determine anomaly severity
        if anomaly_score > p97_normal:
            anomaly_severity = "CRITICAL"
        elif anomaly_score > p90_normal:
            anomaly_severity = "WARNING"
        else:
            anomaly_severity = "NORMAL"
        
        print(f"  ✓ Anomaly severity: {anomaly_severity}")
        
        # Determine RUL status
        if rul_pred > 80:
            rul_status = "NORMAL"
        elif rul_pred > 30:
            rul_status = "WARNING"
        else:
            rul_status = "CRITICAL"
        
        print(f"  ✓ RUL status: {rul_status} ({rul_pred:.1f} cycles)")
        
        # Combined status
        if "CRITICAL" in (anomaly_severity, rul_status):
            combined = "CRITICAL"
        elif "WARNING" in (anomaly_severity, rul_status):
            combined = "WARNING"
        else:
            combined = "NORMAL"
        
        print(f"  ✓ COMBINED STATUS: {combined}")
        
        # Test sensor contribution
        try:
            from utils.sensor_contribution import get_top_contributors
            from utils.severity import get_group_severity
            
            top_sensors = get_top_contributors(raw_matrix, window, iso_scaler)
            print(f"  ✓ Top sensors: {top_sensors}")
            
            group_severity = get_group_severity(raw_matrix, window, iso_scaler)
            print(f"  ✓ Group severity: {group_severity}")
            
        except Exception as e:
            print(f"  ⚠ Sensor contribution test failed: {e}")
        
        INFERENCE_OK = True
        
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        INFERENCE_OK = False
else:
    print(f"  ⚠ Skipping inference (data or features not ready)")
    INFERENCE_OK = False

# ============================================
# 5. SUMMARY
# ============================================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

print(f"\nData Loading: {'✓ PASS' if DATA_LOADED else '✗ FAIL'}")
print(f"Isolation Forest: {'✓ PASS' if models_loaded.get('iso', False) else '✗ FAIL'}")
print(f"LSTM Model: {'✓ PASS' if models_loaded.get('lstm', False) else '✗ FAIL'}")
print(f"Feature Engineering: {'✓ PASS' if FEATURES_OK else '✗ FAIL'}")
print(f"Inference: {'✓ PASS' if INFERENCE_OK else '✗ FAIL'}")

if all([DATA_LOADED, models_loaded.get('iso', False), 
        models_loaded.get('lstm', False), FEATURES_OK]):
    print("\n✅ ALL SYSTEMS GO! You can proceed with the Flask app.")
else:
    print("\n⚠ Some components failed. Fix issues before running Flask app.")