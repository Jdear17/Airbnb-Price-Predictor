import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from model import train_model, predict_price

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Use all available processors
)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    # print MAE, MSE, R²
    mse = mean_squared_error(y_test, preds)
    mae = np.mean(np.abs(y_test - preds))
    r2 = model.score(X_test, y_test)
    print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
    return mse, mae, r2


