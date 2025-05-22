import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load dataset
def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)

# Display basic info about the dataset
def display_data_info(df):
    """
    Display basic information about the dataset.
    """
    print("Data Info:")
    print(df.info())
    print("\nData Description:")
    print(df.describe())

def preprocess_data(X,y):

    categorical_features = ['room_type', 'neighbourhood']
    numerical_features = ['availability_365', 'reviews_per_month', 'calculated_host_listings_count', 'distance_to_center']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # keep numerical columns as-is
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', get_rf_model())
    ])
    
    return pipeline, X_train, X_test, y_train, y_test
