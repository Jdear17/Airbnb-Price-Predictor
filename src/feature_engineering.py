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


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    london_center = (51.494720, -0.135278)


    # Calculate distance to city center
    df['distance_to_center'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), london_center).km,
        axis=1
    )
    # Keep the original 'room_type' column and one-hot encode categorical columns
    X = df[[
        'calculated_host_listings_count',
        'availability_365',
        'reviews_per_month',
        'distance_to_center',
        'room_type',
        'neighbourhood'
    ]]

    y = df['price']

    return X, y
