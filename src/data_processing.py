def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, target_column, categorical_features, numerical_features):
    """
    Preprocess the data by handling categorical and numerical features.
    """
    # Splitting the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting into train and test sets