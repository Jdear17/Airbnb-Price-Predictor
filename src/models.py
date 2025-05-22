import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def get_rf_model():
    return RandomForestRegressor(
        max_depth=10,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

def get_xgb_model():
    return XGBRegressor(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )



def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    # print MAE, MSE, R²
    mse = mean_squared_error(y_test, preds)
    mae = np.mean(np.abs(y_test - preds))
    r2 = model.score(X_test, y_test)
    print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
    return mse, mae, r2


