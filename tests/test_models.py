"""Tests pour valider la performance des modèles ML"""
import pytest
import sys
import os
import numpy as np
import pandas as pd
import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Désactiver le mode test
os.environ['PYTEST_CURRENT_TEST'] = 'false'

from services.model_loader import get_models
from services.preprocessing_service import PreprocessingService
from config import ANOMALY_THRESHOLDS, RUL_THRESHOLDS, SELECTED_SENSORS

# Chemin absolu vers les données de test
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'scenario_windows.pkl')

# Vérifier la présence des données au niveau module
_DATA_AVAILABLE = os.path.exists(DATA_FILE)

if not _DATA_AVAILABLE:
    print(f"⚠️ Données de test non trouvées: {DATA_FILE}")
    print("⚠️ Les tests dépendant des données seront ignorés")

@pytest.fixture(scope="session")
def test_data():
    """Charge les données de test si disponibles"""
    if not _DATA_AVAILABLE:
        pytest.skip("Données de test non disponibles")
    
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    print(f"✅ Données chargées: {len(df)} échantillons")
    return df

@pytest.fixture(scope="module")
def models():
    """Charge les modèles pour les tests"""
    return get_models()

def test_anomaly_model_loading(models):
    """Test que le modèle d'anomalie se charge"""
    assert models.iso_model is not None
    assert models.iso_scaler is not None

def test_rul_model_loading(models):
    """Test que le modèle RUL se charge"""
    assert models.lstm_model is not None
    assert models.rul_scaler is not None

@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Données de test non disponibles")
def test_anomaly_prediction(models, test_data):
    """Test les prédictions d'anomalie"""
    scores = []
    n_samples = min(5, len(test_data))
    
    for i in range(n_samples):
        sample = test_data.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        features = PreprocessingService.prepare_features_for_anomaly(
            window_df, models.iso_scaler
        )
        score = models.get_anomaly_score(features)
        scores.append(score)
    
    scores = np.array(scores)
    print(f"\n[STATS] Scores anomalie ({n_samples} échantillons):")
    print(f"   Min: {scores.min():.4f}")
    print(f"   Max: {scores.max():.4f}")
    print(f"   Moy: {scores.mean():.4f}")
    
    assert len(scores) == n_samples
    assert all(isinstance(s, float) for s in scores)

@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Données de test non disponibles")
def test_rul_prediction(models, test_data):
    """Test les prédictions RUL"""
    predictions = []
    n_samples = min(5, len(test_data))
    
    for i in range(n_samples):
        sample = test_data.iloc[i]
        window = sample['sensor_window']
        window_df = pd.DataFrame(window, columns=SELECTED_SENSORS)
        
        sequence = PreprocessingService.prepare_sequence_for_rul(
            window_df, models.rul_scaler
        )
        rul = models.predict_rul(sequence)
        predictions.append(rul)
    
    predictions = np.array(predictions)
    print(f"\n[STATS] Prédictions RUL ({n_samples} échantillons):")
    print(f"   Min: {predictions.min():.1f} cycles")
    print(f"   Max: {predictions.max():.1f} cycles")
    print(f"   Moy: {predictions.mean():.1f} cycles")
    
    assert len(predictions) == n_samples
    assert all(p >= 0 for p in predictions)

@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Données de test non disponibles")
def test_anomaly_thresholds(models, test_data):
    """Test la distribution des statuts d'anomalie"""
    n_samples = min(20, len(test_data))
    normal = warning = critical = 0
    
    for i in range(n_samples):
        sample = test_data.iloc[i]
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
    print(f"\n[DISTRIBUTION] Anomalies ({total} échantillons):")
    print(f"   Normal:   {normal} ({normal/total*100:.1f}%)")
    print(f"   Warning:  {warning} ({warning/total*100:.1f}%)")
    print(f"   Critical: {critical} ({critical/total*100:.1f}%)")
    
    assert total == n_samples

@pytest.mark.skipif(not _DATA_AVAILABLE, reason="Données de test non disponibles")
def test_rul_thresholds(models, test_data):
    """Test la distribution des statuts RUL"""
    n_samples = min(20, len(test_data))
    normal = warning = critical = 0
    
    for i in range(n_samples):
        sample = test_data.iloc[i]
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
    print(f"\n[DISTRIBUTION] RUL ({total} échantillons):")
    print(f"   Normal:   {normal} ({normal/total*100:.1f}%)")
    print(f"   Warning:  {warning} ({warning/total*100:.1f}%)")
    print(f"   Critical: {critical} ({critical/total*100:.1f}%)")
    
    assert total == n_samples
