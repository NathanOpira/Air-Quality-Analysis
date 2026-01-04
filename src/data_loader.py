"""Data loading helpers for the project.

Provides a simple `load_processed_data` function that returns the cleaned
DataFrame used by the notebooks and analysis modules.

The loader is intentionally conservative: it will try a few sensible
parsing options so it works with common exported CSVs.
"""
from typing import Tuple
import pandas as pd
import os


def load_processed_data(csv_path: str = None) -> pd.DataFrame:
    """Load the processed air quality CSV and return a DataFrame.

    Parameters
    ----------
    csv_path : str, optional
        Path to the processed CSV file. If None, defaults to
        `data/processed/air_quality_cleaned.csv` in the repo root.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a datetime index when possible.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'air_quality_cleaned.csv')
        csv_path = os.path.normpath(csv_path)

    # Try a few parsing strategies
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Processed data not found at {csv_path}") from exc

    # If there's a column named 'date' or 'datetime', parse it
    date_cols = [c for c in df.columns if c.lower() in ('date', 'datetime', 'time')]
    if date_cols:
        try:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.set_index(date_cols[0])
        except Exception:
            pass
    else:
        # If first column looks like datetimes, try parsing index
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        except Exception:
            pass

    return df
