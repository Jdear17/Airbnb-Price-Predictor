import pandas as pd
import numpy as np
from geopy.distance import geodesic

# hadle missing values
def handle_missing_values(df, missing_threshold=0.7):
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])

    df = df.dropna(thresh=len(df) * missing_threshold, axis=1)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

# Filter outliers by room_type
def filter_outliers_by_room_type(df, quantile=0.95):
    percentiles = df.groupby('room_type')['price'].quantile(quantile)
    filtered_df = df[df.apply(lambda row: row['price'] <= percentiles[row['room_type']], axis=1)]
    return filtered_df


def feature_engineering(df: pd.DataFrame, london_center: tuple) -> (pd.DataFrame, pd.Series):
    """
    Perform feature engineering on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing Airbnb data.
        london_center (tuple): Latitude and longitude of the city center.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable (price).
    """

    # Calculate distance to city center
    df['distance_to_center'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), london_center).km,
        axis=1
    )

    # Keep the original 'room_type' column and one-hot encode categorical columns
    df['type'] = df['room_type']
    df = pd.get_dummies(df, columns=['neighbourhood'], drop_first=False)
    df = pd.get_dummies(df, columns=['room_type'], drop_first=False)

    # Define feature and target variables
    X = df[[
        'calculated_host_listings_count',
        'availability_365',
        'reviews_per_month',
        'distance_to_center',
    ] + [col for col in df.columns if 'room_type' in col or 'neighbourhood' in col]]
    y = df['price']

    return X, y
















def add_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for Airbnb price prediction.
    """
    # Extract date-related features
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Create binary features for amenities
    if 'amenities' in data.columns:
        data['has_wifi'] = data['amenities'].str.contains('wifi', case=False, na=False).astype(int)
        data['has_kitchen'] = data['amenities'].str.contains('kitchen', case=False, na=False).astype(int)
        data['has_parking'] = data['amenities'].str.contains('parking', case=False, na=False).astype(int)

    # Calculate room-related features
    if 'bedrooms' in data.columns and 'beds' in data.columns:
        data['beds_per_bedroom'] = data['beds'] / data['bedrooms'].replace(0, np.nan)

    # Log-transform price to reduce skewness
    if 'price' in data.columns:
        data['log_price'] = np.log1p(data['price'])

    # Create interaction terms
    if 'accommodates' in data.columns and 'bathrooms' in data.columns:
        data['accommodates_per_bathroom'] = data['accommodates'] / data['bathrooms'].replace(0, np.nan)

    # Fill missing values (example: replace NaN with 0 for numerical features)
    data.fillna(0, inplace=True)

    return data