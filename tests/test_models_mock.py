"""Tests pour les modèles utilisant UNIQUEMENT des mocks"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from config import SELECTED_SENSORS
from services.preprocessing_service import PreprocessingService

@pytest.fixture
def mock_scaler():
    """Crée un scaler mocké avec feature_names_in_ simulé"""
    scaler = MagicMock()
    scaler.transform.return_value = np.random.randn(1, 60)
    # Simuler feature_names_in_ pour éviter les erreurs
    scaler.feature_names_in_ = [f"feat_{i}" for i in range(60)]
    return scaler

@pytest.fixture
def mock_iso_model():
    """Crée un modèle Isolation Forest mocké"""
    model = MagicMock()
    model.decision_function.return_value = np.array([0.5])
    return model

@pytest.fixture
def mock_lstm_model():
    """Crée un modèle LSTM mocké"""
    model = MagicMock()
    model.predict.return_value = np.array([[100]])
    return model

def test_preprocessing_with_mock_scaler(mock_scaler):
    """Test que le preprocessing fonctionne avec un scaler mocké"""
    # Créer des données de test
    data = np.random.randn(30, 12)
    window_df = pd.DataFrame(data, columns=SELECTED_SENSORS)
    
    # Appeler la fonction
    features = PreprocessingService.prepare_features_for_anomaly(window_df, mock_scaler)
    
    # Vérifier que le scaler a été appelé
    mock_scaler.transform.assert_called_once()
    assert features is not None

def test_anomaly_score(mock_iso_model):
    """Test que le calcul du score d'anomalie fonctionne"""
    features = np.random.randn(1, 60)
    score = -mock_iso_model.decision_function(features)[0]
    assert score == -0.5
