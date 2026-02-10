import pandas as pd

def preprocess_input(df, feature_columns, scaler):
    """
    Takes raw input DataFrame and returns model-ready scaled features.

    Parameters:
    - df: pandas DataFrame (raw input)
    - feature_columns: list of feature names used during training
    - scaler: fitted StandardScaler

    Returns:
    - Scaled numpy array ready for model prediction
    """

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df)

    # Align columns with training features
    df_encoded = df_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale using trained scaler
    df_scaled = scaler.transform(df_encoded)

    return df_scaled
