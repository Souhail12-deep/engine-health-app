import pandas as pd
import numpy as np
import joblib
import os
import sys
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SELECTED_SENSORS, WINDOW_SIZE

def load_models():
    """Load trained models"""
    from services.model_loader import models
    return models

def process_test_data():
    """Process test data and save complete windows for each scenario"""
    
    print("🔍 Loading test data...")
    test_path = os.path.join("data", "test", "test_FD001.txt")
    cols = (["unit", "cycle"] + [f"op{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)])
    test_df = pd.read_csv(test_path, sep="\s+", header=None, names=cols)
    print(f"✓ Raw data loaded: {test_df.shape}")
    
    # Load models
    models = load_models()
    
    # Store results with COMPLETE WINDOWS
    results = []
    
    print("🔄 Processing engines...")
    
    for engine_id in test_df['unit'].unique():
        engine_data = test_df[test_df['unit'] == engine_id].sort_values('cycle').reset_index(drop=True)
        
        # Skip if engine doesn't have enough cycles
        if len(engine_data) < WINDOW_SIZE:
            print(f"  Engine {engine_id}: only {len(engine_data)} cycles, skipping")
            continue
        
        print(f"  Processing Engine {engine_id} with {len(engine_data)} cycles...")
        
        # Process each possible window
        for end_idx in range(WINDOW_SIZE - 1, len(engine_data)):
            start_idx = end_idx - WINDOW_SIZE + 1
            window = engine_data.iloc[start_idx:end_idx + 1].copy()
            
            # Get the last cycle number (for reference)
            cycle = window.iloc[-1]['cycle']
            
            # Extract just the sensor values for this window
            sensor_window = window[SELECTED_SENSORS].values.tolist()
            
            # Prepare features for prediction (to get scores)
            from services.preprocessing_service import PreprocessingService
            anomaly_features = PreprocessingService.prepare_features_for_anomaly(
                window, models.iso_scaler
            )
            rul_sequence = PreprocessingService.prepare_sequence_for_rul(
                window, models.rul_scaler
            )
            
            # Get predictions
            anomaly_score = models.get_anomaly_score(anomaly_features)
            rul_prediction = models.predict_rul(rul_sequence)
            
            # Determine scenario
            from config import ANOMALY_THRESHOLDS, RUL_THRESHOLDS
            if anomaly_score <= ANOMALY_THRESHOLDS["WARNING"] and rul_prediction > RUL_THRESHOLDS["WARNING"]:
                scenario = "normal"
            elif anomaly_score > ANOMALY_THRESHOLDS["CRITICAL"] or rul_prediction <= RUL_THRESHOLDS["CRITICAL"]:
                scenario = "critical"
            else:
                scenario = "warning"
            
            # Store the COMPLETE WINDOW, not just the cycle number
            results.append({
                'engine_id': int(engine_id),
                'cycle': int(cycle),
                'scenario': scenario,
                'anomaly_score': float(anomaly_score),
                'rul': float(rul_prediction),
                'sensor_window': sensor_window,  # Store the entire window!
                'window_size': len(sensor_window)
            })
    
    # Convert to DataFrame
    df_all = pd.DataFrame(results)
    print(f"\n✅ Processed {len(results)} total samples")
    
    # Get counts per scenario
    for scenario in ['normal', 'warning', 'critical']:
        count = len(df_all[df_all['scenario'] == scenario])
        print(f"   {scenario.capitalize()}: {count} samples")
    
    # Save samples for each scenario (50 each)
    samples_per_scenario = 50
    final_samples = []
    
    for scenario in ['normal', 'warning', 'critical']:
        scenario_data = df_all[df_all['scenario'] == scenario]
        if len(scenario_data) > 0:
            n_samples = min(samples_per_scenario, len(scenario_data))
            sampled = scenario_data.sample(n_samples, random_state=42)
            final_samples.append(sampled)
    
    # Combine and save
    if final_samples:
        final_df = pd.concat(final_samples).reset_index(drop=True)
        
        # Save as pickle (preserves complex objects like lists)
        pickle_path = os.path.join("data", "test", "scenario_windows.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(final_df.to_dict('records'), f)
        
        # Also save as CSV for inspection (sensor_window will be string)
        final_df['sensor_window_str'] = final_df['sensor_window'].apply(str)
        csv_path = os.path.join("data", "test", "scenario_windows.csv")
        final_df[['engine_id', 'cycle', 'scenario', 'anomaly_score', 'rul', 'sensor_window_str']].to_csv(csv_path, index=False)
        
        print(f"\n✓ Saved {len(final_df)} samples to {pickle_path}")
        print(f"✓ Also saved CSV version to {csv_path}")
        
        return final_df
    else:
        print("❌ No samples found!")
        return None

if __name__ == "__main__":
    process_test_data()