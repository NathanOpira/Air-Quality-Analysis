import pandas as pd
import numpy as np

def create_temporal_features(df):
    """Create temporal features from datetime index"""
    df_temp = df.copy()

    if isinstance(df_temp.index, pd.DatetimeIndex):
        # Basic time features
        df_temp['hour'] = df_temp.index.hour
        df_temp['day'] = df_temp.index.day
        df_temp['month'] = df_temp.index.month
        df_temp['year'] = df_temp.index.year
        df_temp['dayofweek'] = df_temp.index.dayofweek
        df_temp['dayofyear'] = df_temp.index.dayofyear

        # Cyclic encoding
        df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
        df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)
        df_temp['dow_sin'] = np.sin(2 * np.pi * df_temp['dayofweek'] / 7)
        df_temp['dow_cos'] = np.cos(2 * np.pi * df_temp['dayofweek'] / 7)

        # Time of day categories
        df_temp['is_night'] = ((df_temp['hour'] >= 0) & (df_temp['hour'] < 6)).astype(int)
        df_temp['is_morning'] = ((df_temp['hour'] >= 6) & (df_temp['hour'] < 12)).astype(int)
        df_temp['is_afternoon'] = ((df_temp['hour'] >= 12) & (df_temp['hour'] < 18)).astype(int)
        df_temp['is_evening'] = ((df_temp['hour'] >= 18) & (df_temp['hour'] < 24)).astype(int)

        # Weekend indicator
        df_temp['is_weekend'] = (df_temp['dayofweek'] >= 5).astype(int)

    return df_temp


def create_pollution_features(df):
    """Create pollution-specific features"""
    df_feat = df.copy()

    # Find pollutant columns
    pollutant_patterns = ['CO', 'NO', 'NOx', 'NO2', 'O3', 'SO2', 'PM', 'BEN', 'RH', 'TEMP']
    pollutant_cols = []

    for col in df_feat.columns:
        for pattern in pollutant_patterns:
            if pattern in col.upper():
                pollutant_cols.append(col)
                break

    if pollutant_cols:
        # Ratios between pollutants
        co_cols = [col for col in pollutant_cols if 'CO' in col.upper()]
        nox_cols = [col for col in pollutant_cols if 'NO' in col.upper()]

        if co_cols and nox_cols:
            co_col = co_cols[0]
            nox_col = nox_cols[0]
            df_feat[f'{co_col}_to_{nox_col}'] = df_feat[co_col] / (df_feat[nox_col] + 1e-6)

        # Rate of change
        for col in pollutant_cols[:3]:
            df_feat[f'{col}_diff'] = df_feat[col].diff()

        # Rolling statistics
        for col in pollutant_cols[:2]:
            df_feat[f'{col}_rolling_mean_6h'] = df_feat[col].rolling(window=6, min_periods=1).mean()

    return df_feat


def get_feature_matrix(df):
    """Extract feature matrix and names"""
    df_features = df.copy()

    # Fill any remaining NaNs
    for col in df_features.columns:
        if df_features[col].isnull().sum() > 0:
            df_features[col] = df_features[col].fillna(df_features[col].mean())

    feature_names = df_features.columns.tolist()
    return df_features, feature_names