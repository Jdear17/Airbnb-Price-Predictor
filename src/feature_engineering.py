import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from typing import Tuple
from src.config import config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def calculate_distance_to_center(df: pd.DataFrame) -> pd.Series:
    """
    Calculate distance from each listing to the city center.
    
    Args:
        df (pd.DataFrame): Input dataframe with latitude and longitude columns
        
    Returns:
        pd.Series: Series containing distances to center
    """
    # London city center coordinates
    center_coords = (51.5074, -0.1278)
    
    distances = df.apply(
        lambda row: geodesic(
            (row['latitude'], row['longitude']),
            center_coords
        ).kilometers,
        axis=1
    )
    return distances

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the last_review column.
    
    Args:
        df (pd.DataFrame): Input dataframe with last_review column
        
    Returns:
        pd.DataFrame: DataFrame with new time features
    """
    df = df.copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Days since last review
    df['days_since_review'] = (pd.Timestamp.now() - df['last_review']).dt.days
    
    # Month of last review
    df['review_month'] = df['last_review'].dt.month
    
    # Season of last review
    df['review_season'] = df['last_review'].dt.month % 12 // 3 + 1
    
    return df

def create_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to the host.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with new host features
    """
    df = df.copy()
    
    # Host experience (days since first review)
    df['host_experience'] = (pd.Timestamp.now() - pd.to_datetime(df['last_review'])).dt.days
    
    # Reviews per listing
    df['reviews_per_listing'] = df['number_of_reviews'] / df['calculated_host_listings_count']
    
    # Host response rate (if available)
    if 'host_response_rate' in df.columns:
        df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype('float') / 100
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df = df.copy()
    
    # Fill missing reviews_per_month with 0
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # Fill missing last_review with a date far in the past
    df['last_review'] = df['last_review'].fillna('2000-01-01')
    
    # Fill missing host_is_superhost with False
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].fillna('f')
    
    return df

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter outliers based on price and other numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df = df.copy()
    
    # Filter price outliers
    price_mean = df['price'].mean()
    price_std = df['price'].std()
    price_threshold = config.data.outlier_threshold
    
    df = df[
        (df['price'] > price_mean - price_threshold * price_std) &
        (df['price'] < price_mean + price_threshold * price_std)
    ]
    
    # Filter other numerical features
    for feature in config.data.numerical_features:
        if feature in df.columns:
            mean = df[feature].mean()
            std = df[feature].std()
            df = df[
                (df[feature] > mean - price_threshold * std) &
                (df[feature] < mean + price_threshold * std)
            ]
    
    return df

def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform feature engineering on the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target
    """
    logger.info("Starting feature engineering...")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create new features
    df['distance_to_center'] = calculate_distance_to_center(df)
    df = create_time_features(df)
    df = create_host_features(df)
    
    # Filter outliers
    df = filter_outliers(df)
    
    # Select features
    features = config.data.categorical_features + config.data.numerical_features
    X = df[features]
    y = df[config.data.target_column]
    
    logger.info(f"Feature engineering complete. Final shape: {X.shape}")
    
    return X, y
