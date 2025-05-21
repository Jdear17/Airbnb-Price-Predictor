import pandas as pd
import numpy as np

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


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    london_center = (51.494720, -0.135278)

    """
    Perform feature engineering on the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing Airbnb data.

    Returns:
        tuple: Feature matrix (X) and target variable (y).
    """

    # Calculate distance to city center
    df['distance_to_center'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), london_center).km,
        axis=1
    )
    london_center = (51.494720, -0.135278)
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
