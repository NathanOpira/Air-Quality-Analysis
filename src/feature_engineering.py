"""
features.py
Feature engineering for air quality analysis.
Focus on geometric and temporal features for distance analysis.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_temporal_features(df):
    """
    Create cyclic temporal features from datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features added
    """
    df_feat = df.copy()

    if df_feat.index.inferred_type == 'datetime64':
        # Hour of day (cyclic)
        df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat.index.hour / 24)
        df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat.index.hour / 24)

        # Day of week (cyclic)
        df_feat['day_sin'] = np.sin(2 * np.pi * df_feat.index.dayofweek / 7)
        df_feat['day_cos'] = np.cos(2 * np.pi * df_feat.index.dayofweek / 7)

        # Month (cyclic)
        df_feat['month_sin'] = np.sin(2 * np.pi * df_feat.index.month / 12)
        df_feat['month_cos'] = np.cos(2 * np.pi * df_feat.index.month / 12)

        # Weekend flag
        df_feat['is_weekend'] = (df_feat.index.dayofweek >= 5).astype(int)

        # Time of day categories
        df_feat['is_night'] = ((df_feat.index.hour >= 0) & (df_feat.index.hour < 6)).astype(int)
        df_feat['is_morning'] = ((df_feat.index.hour >= 6) & (df_feat.index.hour < 12)).astype(int)
        df_feat['is_afternoon'] = ((df_feat.index.hour >= 12) & (df_feat.index.hour < 18)).astype(int)
        df_feat['is_evening'] = ((df_feat.index.hour >= 18)).astype(int)

    return df_feat

def create_pollution_features(df):
    """
    Create pollution-specific features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pollution columns

    Returns
    -------
    pd.DataFrame
        DataFrame with pollution features added
    """
    df_feat = df.copy()

    # Pollutant ratios (combustion signatures)
    if 'CO(GT)' in df_feat.columns and 'NOx(GT)' in df_feat.columns:
        df_feat['CO_to_NOx'] = df_feat['CO(GT)'] / (df_feat['NOx(GT)'] + 1e-10)

    if 'NO2(GT)' in df_feat.columns and 'NOx(GT)' in df_feat.columns:
        df_feat['NO2_fraction'] = df_feat['NO2(GT)'] / (df_feat['NOx(GT)'] + 1e-10)

    # Meteorological interactions
    if 'T' in df_feat.columns and 'RH' in df_feat.columns:
        df_feat['heat_index'] = 0.5 * (df_feat['T'] + 61.0 + (df_feat['T'] - 68.0) * 1.2 + df_feat['RH'] * 0.094)

    # Rate of change (1-hour difference)
    pollution_cols = [col for col in df.columns if '(GT)' in col]
    for col in pollution_cols[:5]:  # First 5 pollutants
        if col in df_feat.columns:
            df_feat[f'{col}_diff'] = df_feat[col].diff().fillna(0)

    return df_feat

def normalize_features(df, feature_columns):
    """
    Normalize features for distance analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    feature_columns : list
        Columns to normalize

    Returns
    -------
    np.ndarray
        Normalized feature matrix
    StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(df[feature_columns])
    return X_normalized, scaler

def get_feature_matrix(df):
    """
    Create complete feature matrix for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame

    Returns
    -------
    pd.DataFrame
        Feature matrix
    list
        Feature names
    """
    # Add temporal features
    df_feat = create_temporal_features(df)

    # Add pollution features
    df_feat = create_pollution_features(df_feat)

    # Select features for analysis
    pollution_features = [col for col in df.columns if '(GT)' in col or col in ['T', 'RH', 'AH']]
    temporal_features = [col for col in df_feat.columns if col not in df.columns]

    all_features = pollution_features + temporal_features

    # Keep only columns that exist
    available_features = [f for f in all_features if f in df_feat.columns]

    return df_feat[available_features], available_features