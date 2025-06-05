from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
from src.config import config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def get_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessor for numerical and categorical features.
    
    Returns:
        ColumnTransformer: Configured preprocessor
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config.data.numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config.data.categorical_features)
        ]
    )
    return preprocessor

def get_rf_model() -> RandomForestRegressor:
    """
    Create a Random Forest model with configured parameters.
    
    Returns:
        RandomForestRegressor: Configured Random Forest model
    """
    return RandomForestRegressor(
        **config.model.rf_params,
        random_state=config.model.random_state
    )

def get_xgb_model() -> XGBRegressor:
    """
    Create an XGBoost model with configured parameters.
    
    Returns:
        XGBRegressor: Configured XGBoost model
    """
    return XGBRegressor(
        **config.model.xgb_params,
        random_state=config.model.random_state
    )

def create_model_pipeline(model) -> Pipeline:
    """
    Create a pipeline with preprocessor and model.
    
    Args:
        model: The model to use in the pipeline
        
    Returns:
        Pipeline: Configured pipeline
    """
    return Pipeline([
        ('preprocessor', get_preprocessor()),
        ('model', model)
    ])

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train the best model using cross-validation.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        Tuple[Pipeline, Dict[str, float]]: Best model and its metrics
    """
    logger.info("Starting model training...")
    
    # Create models
    models = {
        'random_forest': get_rf_model(),
        'xgboost': get_xgb_model()
    }
    
    best_score = float('-inf')
    best_model = None
    best_metrics = {}
    
    # Cross-validation
    cv = KFold(n_splits=config.model.cv_folds, shuffle=True, random_state=config.model.random_state)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        pipeline = create_model_pipeline(model)
        
        # Perform cross-validation
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        logger.info(f"{name} CV R² score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_metrics = {
                'mean_r2': mean_score,
                'std_r2': std_score,
                'model_name': name
            }
    
    # Train the best model on the full dataset
    logger.info(f"Training final model ({best_metrics['model_name']})...")
    best_model.fit(X, y)
    
    # Save the model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, f"{best_metrics['model_name']}_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return best_model, best_metrics

def load_model(model_name: str) -> Pipeline:
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        Pipeline: Loaded model
    """
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)


