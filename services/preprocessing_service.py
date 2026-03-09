import pandas as pd
import numpy as np
from config import SELECTED_SENSORS, MODEL_SENSOR_NAMES, WINDOW_SIZE, RAW_TO_MODEL_MAP

class PreprocessingService:
    
    @staticmethod
    def get_window_from_test(engine_id, cycle, test_df):
        """
        Extract a window of WINDOW_SIZE cycles ending at the given cycle
        """
        # Get all data for this engine
        engine_data = test_df[test_df['unit'] == engine_id].sort_values('cycle').reset_index(drop=True)
        
        # Find the row with this cycle
        matching_rows = engine_data[engine_data['cycle'] == cycle]
        
        if len(matching_rows) == 0:
            print(f"  No cycle {cycle} found for engine {engine_id}")
            return None
        
        # Get the position (index) of this cycle
        cycle_position = matching_rows.index[0]
        
        # Need at least WINDOW_SIZE cycles before and including this one
        if cycle_position < WINDOW_SIZE - 1:
            print(f"  Engine {engine_id} at cycle {cycle}: only {cycle_position+1} cycles available, need {WINDOW_SIZE}")
            return None
        
        # Get window of last WINDOW_SIZE cycles
        start_idx = cycle_position - WINDOW_SIZE + 1
        window = engine_data.iloc[start_idx:cycle_position + 1].copy()
        
        if len(window) != WINDOW_SIZE:
            print(f"  Window size mismatch: got {len(window)}, expected {WINDOW_SIZE}")
            return None
        
        return window.reset_index(drop=True)
    
    @staticmethod
    def prepare_features_for_anomaly(window_df, iso_scaler):
        """
        Prepare features for anomaly detection model
        """
        # Rename sensors to match training format
        df = window_df.rename(columns=RAW_TO_MODEL_MAP)
        
        # Calculate engineered features
        for s in MODEL_SENSOR_NAMES:
            # Mean
            df[f"{s}_mean"] = df[s].rolling(WINDOW_SIZE, min_periods=1).mean()
            
            # Std
            df[f"{s}_std"] = df[s].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
            
            # Delta
            df[f"{s}_delta"] = df[s].diff().fillna(0)
            
            # Slope
            slopes = []
            values = df[s].values
            for i in range(len(values)):
                start = max(0, i - WINDOW_SIZE + 1)
                y = values[start:i+1]
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                else:
                    slope = 0
                slopes.append(slope)
            df[f"{s}_slope"] = slopes
        
        # Get last row features
        feature_row = df.iloc[-1:].copy()
        
        # Select only features that scaler expects
        expected_features = iso_scaler.feature_names_in_
        available_features = [col for col in expected_features if col in feature_row.columns]
        feature_row = feature_row[available_features]
        
        # Scale
        scaled = iso_scaler.transform(feature_row)
        
        return scaled
    
    @staticmethod
    def prepare_sequence_for_rul(window_df, rul_scaler):
        """
        Prepare sequence for RUL prediction model
        """
        # Select sensors in correct order
        sensor_data = window_df[SELECTED_SENSORS].copy()
        
        # Ensure columns match scaler expectations
        sensor_data = sensor_data[rul_scaler.feature_names_in_]
        
        # Scale
        scaled = rul_scaler.transform(sensor_data)
        
        # Reshape for LSTM: (1, window_size, n_features)
        sequence = scaled.reshape(1, WINDOW_SIZE, len(SELECTED_SENSORS))
        
        return sequence