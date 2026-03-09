import numpy as np
import pandas as pd
from config import SELECTED_SENSORS, SENSOR_GROUPS

def calculate_sensor_contributions(raw_window, anomaly_score_value=None):
    """
    Calculate which sensors contribute most to the anomaly
    Returns top 3 sensors and their group severity
    """
    if raw_window is None or len(raw_window) == 0:
        return [], {}
    
    # Calculate variance/change in each sensor over the window
    contributions = {}
    
    for i, sensor in enumerate(SELECTED_SENSORS):
        # Use standard deviation as measure of variation
        sensor_data = raw_window[:, i] if hasattr(raw_window, 'shape') else raw_window[sensor].values
        
        if len(sensor_data) > 1:
            # Calculate trend strength (absolute slope)
            x = np.arange(len(sensor_data))
            slope = np.polyfit(x, sensor_data, 1)[0]
            contributions[sensor] = abs(slope) * np.std(sensor_data)
        else:
            contributions[sensor] = 0
    
    # Get top 3 sensors
    top_sensors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    top_sensor_names = [s[0] for s in top_sensors]
    
    # Calculate group severity
    group_severity = {}
    for group, sensors in SENSOR_GROUPS.items():
        group_scores = [contributions.get(s, 0) for s in sensors if s in contributions]
        if group_scores:
            avg_score = np.mean(group_scores)
            if avg_score > 0.5:
                group_severity[group] = "HIGH"
            elif avg_score > 0.2:
                group_severity[group] = "MEDIUM"
            else:
                group_severity[group] = "LOW"
        else:
            group_severity[group] = "LOW"
    
    return top_sensor_names, group_severity