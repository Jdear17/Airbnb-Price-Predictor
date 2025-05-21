import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the given model on the test dataset.

    Parameters:
        model: Trained model to evaluate.
        X_test: Features of the test dataset.
        y_test: True labels of the test dataset.

    Returns:
        A dictionary containing evaluation metrics.
    """
    # Predict the prices
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print metrics
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

# Example usage:
# Assuming you have a trained model, X_test, and y_test
# from sklearn.externals import joblib
# model = joblib.load('path_to_your_model.pkl')
# metrics = evaluate_model(model, X_test, y_test)