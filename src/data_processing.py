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

def preprocess_data(df, target_column, categorical_features, numerical_features):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    return pipeline, X_train, X_test, y_train, y_test
