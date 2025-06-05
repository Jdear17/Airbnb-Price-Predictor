from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1
    }
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1
    }

@dataclass
class DataConfig:
    categorical_features: List[str] = [
        'room_type',
        'neighbourhood',
        'host_is_superhost',
        'instant_bookable'
    ]
    
    numerical_features: List[str] = [
        'availability_365',
        'reviews_per_month',
        'calculated_host_listings_count',
        'distance_to_center',
        'minimum_nights',
        'number_of_reviews',
        'number_of_reviews_ltm'
    ]
    
    target_column: str = 'price'
    
    # Outlier detection
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    
    # Paths
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, 'raw', 'listingss.csv')
    PROCESSED_DATA_PATH: str = os.path.join(DATA_DIR, 'processed', 'processed_listings.csv')
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'app.log')

config = Config() 