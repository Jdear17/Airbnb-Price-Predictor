import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from model import train_model, predict_price

def test_train_model():
    # Sample training data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([100, 200, 300])
    
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestRegressor), "Model should be a RandomForestRegressor"

def test_predict_price():
    # Sample data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([100, 200, 300])
    X_test = np.array([[2, 3]])
    
    model = train_model(X_train, y_train)
    predictions = predict_price(model, X_test)
    
    assert len(predictions) == len(X_test), "Number of predictions should match number of test samples"
    assert predictions[0] > 0, "Predicted price should be positive"

def test_model_performance():
    # Sample data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([100, 200, 300])
    X_test = np.array([[2, 3]])
    y_test = np.array([150])
    
    model = train_model(X_train, y_train)
    predictions = predict_price(model, X_test)
    mse = mean_squared_error(y_test, predictions)
    
    assert mse >= 0, "Mean Squared Error should be non-negative"