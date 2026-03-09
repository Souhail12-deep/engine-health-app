from config import RUL_THRESHOLDS, ANOMALY_THRESHOLDS

def determine_status(rul_value, anomaly_score):
    """
    Determine final status based on both RUL and anomaly score
    """
    # RUL-based status
    if rul_value <= RUL_THRESHOLDS["CRITICAL"]:
        rul_status = "CRITICAL"
    elif rul_value <= RUL_THRESHOLDS["WARNING"]:
        rul_status = "WARNING"
    else:
        rul_status = "NORMAL"
    
    # Anomaly-based status
    if anomaly_score > ANOMALY_THRESHOLDS["CRITICAL"]:
        anomaly_status = "CRITICAL"
    elif anomaly_score > ANOMALY_THRESHOLDS["WARNING"]:
        anomaly_status = "WARNING"
    else:
        anomaly_status = "NORMAL"
    
    # Combined status (most severe wins)
    status_priority = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    
    combined = max(rul_status, anomaly_status, key=lambda x: status_priority[x])
    
    return combined, rul_status, anomaly_status