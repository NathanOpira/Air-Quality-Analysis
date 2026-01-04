"""
data_loader.py
Professional data loading and preprocessing for UCI Air Quality Dataset.
Handles European formatting, sentinel values, and quality control.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(filepath='data/raw/AirQualityUCI.csv'):
    """
    Load the raw UCI Air Quality dataset with correct European formatting.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with proper parsing
    """
    # Read with European settings
    df = pd.read_csv(
        filepath,
        sep=';',
        decimal=',',
        na_values=[-200, -200.0]
    )

    # Merge date and time
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format='%d/%m/%Y %H.%M.%S',
        errors='coerce'
    )

    # Set as index and drop original columns
    df = df.set_index('datetime')
    df = df.drop(columns=['Date', 'Time'])

    return df

def clean_data(df, missing_threshold=0.3):
    """
    Clean the air quality data with intelligent handling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame
    missing_threshold : float
        Remove columns with >threshold% missing values

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    dict
        Metadata about cleaning process
    """
    # Remove columns with too many missing values
    missing_ratio = df.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    df_clean = df.drop(columns=columns_to_drop)

    # Time-aware interpolation for time-series
    df_filled = df_clean.interpolate(method='time', limit=6)
    df_filled = df_filled.ffill().bfill()  # Handle edges

    # Remove any remaining NaN rows
    df_final = df_filled.dropna()

    # Generate metadata
    metadata = {
        'original_shape': df.shape,
        'final_shape': df_final.shape,
        'removed_columns': columns_to_drop,
        'kept_columns': df_final.columns.tolist(),
        'date_range': [df_final.index.min(), df_final.index.max()],
        'total_samples': len(df_final)
    }

    return df_final, metadata

def save_processed_data(df, filepath='data/processed/air_quality_cleaned.csv'):
    """Save processed data to CSV."""
    df.to_csv(filepath)
    print(f" Data saved to {filepath}")

def run_data_pipeline():
    """Complete pipeline: load → clean → save."""
    print(" Starting data pipeline...")

    # Load
    df_raw = load_raw_data()
    print(f" Loaded raw data: {df_raw.shape}")

    # Clean
    df_clean, meta = clean_data(df_raw)
    print(f" Cleaned data: {df_clean.shape}")
    print(f" Date range: {meta['date_range'][0]} to {meta['date_range'][1]}")

    # Save
    save_processed_data(df_clean)

    return df_clean, meta