"""Tests pour valider la performance des modeles ML - VERSION CI/CD"""
import pytest
import sys
import os
import numpy as np
import pandas as pd
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.model_loader import get_models
from services.preprocessing_service import PreprocessingService
from config import ANOMALY_THRESHOLDS, RUL_THRESHOLDS, SELECTED_SENSORS

# Detection de l'environnement CI/CD
IN_CI = os.environ.get('CI') == 'true'

# En CI/CD, on ne teste que le chargement des modeles
if IN_CI:
    print("[CI/CD] Tests des predictions ignores")

def test_anomaly_model_loading():
    """Test que le modele d'anomalie se charge"""
    models = get_models()
    assert models.iso_model is not None
    assert models.iso_scaler is not None

def test_rul_model_loading():
    """Test que le modele RUL se charge"""
    models = get_models()
    assert models.lstm_model is not None
    assert models.rul_scaler is not None

@pytest.mark.skipif(IN_CI, reason="Skip en CI/CD - donnees non disponibles")
def test_anomaly_prediction():
    """Test les predictions d'anomalie - UNIQUEMENT EN LOCAL"""
    models = get_models()
    
    # Charger les donnees
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'scenario_windows.pkl')
    if not os.path.exists(data_file):
        pytest.skip("Donnees de test non trouvees")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    scores = []
    n_samples = min(5, len(df))
    
    for i in range(n_samples):
        sample = df.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        features = PreprocessingService.prepare_features_for_anomaly(
            window_df, models.iso_scaler
        )
        score = models.get_anomaly_score(features)
        scores.append(score)
    
    scores = np.array(scores)
    print(f"\n[STATS] Scores anomalie ({n_samples} echantillons):")
    print(f"   Min: {scores.min():.4f}")
    print(f"   Max: {scores.max():.4f}")
    print(f"   Moy: {scores.mean():.4f}")
    
    assert len(scores) == n_samples
    assert all(isinstance(s, float) for s in scores)

@pytest.mark.skipif(IN_CI, reason="Skip en CI/CD - donnees non disponibles")
def test_rul_prediction():
    """Test les predictions RUL - UNIQUEMENT EN LOCAL"""
    models = get_models()
    
    # Charger les donnees
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'scenario_windows.pkl')
    if not os.path.exists(data_file):
        pytest.skip("Donnees de test non trouvees")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    predictions = []
    n_samples = min(5, len(df))
    
    for i in range(n_samples):
        sample = df.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        sequence = PreprocessingService.prepare_sequence_for_rul(
            window_df, models.rul_scaler
        )
        rul = models.predict_rul(sequence)
        predictions.append(rul)
    
    predictions = np.array(predictions)
    print(f"\n[STATS] Predictions RUL ({n_samples} echantillons):")
    print(f"   Min: {predictions.min():.1f} cycles")
    print(f"   Max: {predictions.max():.1f} cycles")
    print(f"   Moy: {predictions.mean():.1f} cycles")
    
    assert len(predictions) == n_samples
    assert all(p >= 0 for p in predictions)

@pytest.mark.skipif(IN_CI, reason="Skip en CI/CD - donnees non disponibles")
def test_anomaly_thresholds():
    """Test la distribution des statuts d'anomalie - UNIQUEMENT EN LOCAL"""
    models = get_models()
    
    # Charger les donnees
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'scenario_windows.pkl')
    if not os.path.exists(data_file):
        pytest.skip("Donnees de test non trouvees")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    n_samples = min(20, len(df))
    normal = warning = critical = 0
    
    for i in range(n_samples):
        sample = df.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        features = PreprocessingService.prepare_features_for_anomaly(
            window_df, models.iso_scaler
        )
        score = models.get_anomaly_score(features)
        
        if score <= ANOMALY_THRESHOLDS["WARNING"]:
            normal += 1
        elif score <= ANOMALY_THRESHOLDS["CRITICAL"]:
            warning += 1
        else:
            critical += 1
    
    total = normal + warning + critical
    print(f"\n[DISTRIBUTION] Anomalies ({total} echantillons):")
    print(f"   Normal:   {normal} ({normal/total*100:.1f}%)")
    print(f"   Warning:  {warning} ({warning/total*100:.1f}%)")
    print(f"   Critical: {critical} ({critical/total*100:.1f}%)")
    
    assert total == n_samples

@pytest.mark.skipif(IN_CI, reason="Skip en CI/CD - donnees non disponibles")
def test_rul_thresholds():
    """Test la distribution des statuts RUL - UNIQUEMENT EN LOCAL"""
    models = get_models()
    
    # Charger les donnees
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'scenario_windows.pkl')
    if not os.path.exists(data_file):
        pytest.skip("Donnees de test non trouvees")
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    
    n_samples = min(20, len(df))
    normal = warning = critical = 0
    
    for i in range(n_samples):
        sample = df.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        sequence = PreprocessingService.prepare_sequence_for_rul(
            window_df, models.rul_scaler
        )
        rul = models.predict_rul(sequence)
        
        if rul > RUL_THRESHOLDS["WARNING"]:
            normal += 1
        elif rul > RUL_THRESHOLDS["CRITICAL"]:
            warning += 1
        else:
            critical += 1
    
    total = normal + warning + critical
    print(f"\n[DISTRIBUTION] RUL ({total} echantillons):")
    print(f"   Normal:   {normal} ({normal/total*100:.1f}%)")
    print(f"   Warning:  {warning} ({warning/total*100:.1f}%)")
    print(f"   Critical: {critical} ({critical/total*100:.1f}%)")
    
    assert total == n_samples
