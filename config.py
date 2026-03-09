import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
ANOMALY_MODEL_PATH = os.path.join(BASE_DIR, "models", "anodet_models", "isolation_forest_latest.pkl")
ANOMALY_SCALER_PATH = os.path.join(BASE_DIR, "models", "anodet_models", "scaler_latest.pkl")
ANOMALY_PARAMS_PATH = os.path.join(BASE_DIR, "models", "anodet_models", "model_params_latest.pkl")

RUL_MODEL_PATH = os.path.join(BASE_DIR, "models", "rul_models", "lstm_rul_latest.keras")
RUL_SCALER_PATH = os.path.join(BASE_DIR, "models", "rul_models", "minmax_scaler_latest.pkl")

# Model parameters
WINDOW_SIZE = 30
SELECTED_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s15", "s17", "s20", "s21"]
MODEL_SENSOR_NAMES = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_15", "s_17", "s_20", "s_21"]
RAW_TO_MODEL_MAP = {raw: model for raw, model in zip(SELECTED_SENSORS, MODEL_SENSOR_NAMES)}

# Thresholds
RUL_THRESHOLDS = {"WARNING": 80, "CRITICAL": 30}
ANOMALY_THRESHOLDS = {"WARNING": -0.0075, "CRITICAL": 0.0051}

# Sensor groups
SENSOR_GROUPS = {
    "temperature": ["s2", "s3", "s4", "s21"],
    "pressure": ["s7", "s11", "s20"],
    "speed": ["s8", "s9"],
    "flow": ["s12", "s15", "s17"]
}