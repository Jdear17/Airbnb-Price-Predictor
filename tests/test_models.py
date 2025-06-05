import pytest
import pandas as pd
import numpy as np
from src.models import get_rf_model, get_xgb_model, create_model_pipeline, train_model
from src.config import config

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n_samples),
        'neighbourhood': np.random.choice(['Kensington', 'Camden', 'Westminster'], n_samples),
        'availability_365': np.random.randint(0, 365, n_samples),
        'reviews_per_month': np.random.uniform(0, 10, n_samples),
        'calculated_host_listings_count': np.random.randint(1, 10, n_samples),
        'distance_to_center': np.random.uniform(0, 20, n_samples),
        'price': np.random.uniform(50, 500, n_samples)
    }
    
    X = pd.DataFrame(data)
    y = X.pop('price')
    
    return X, y

def test_rf_model_creation():
    """Test Random Forest model creation."""
    model = get_rf_model()
    assert model is not None
    assert model.n_estimators == config.model.rf_params['n_estimators']
    assert model.max_depth == config.model.rf_params['max_depth']

def test_xgb_model_creation():
    """Test XGBoost model creation."""
    model = get_xgb_model()
    assert model is not None
    assert model.n_estimators == config.model.xgb_params['n_estimators']
    assert model.max_depth == config.model.xgb_params['max_depth']

def test_model_pipeline_creation():
    """Test model pipeline creation."""
    model = get_rf_model()
    pipeline = create_model_pipeline(model)
    assert pipeline is not None
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'model'

def test_model_training(sample_data):
    """Test model training with sample data."""
    X, y = sample_data
    model, metrics = train_model(X, y)
    
    assert model is not None
    assert 'mean_r2' in metrics
    assert 'std_r2' in metrics
    assert 'model_name' in metrics
    assert metrics['mean_r2'] > 0  # R² should be positive for a good model 