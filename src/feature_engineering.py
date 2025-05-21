import pandas as pd
import numpy as np

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