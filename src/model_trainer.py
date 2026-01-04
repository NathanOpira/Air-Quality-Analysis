"""
model_trainer.py
Machine learning model training and evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
from datetime import datetime

def prepare_target(df, target_col='CO(GT)', lag=1):
    """
    Prepare target variable for time series prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Column to predict
    lag : int
        Number of hours to predict ahead

    Returns
    -------
    pd.DataFrame
        Features
    pd.Series
        Target
    """
    X = df.copy()
    y = df[target_col].shift(-lag).iloc[:-lag]
    X = X.iloc[:-lag]

    return X, y

def train_models(X, y, test_size=0.2, random_state=42):
    """
    Train multiple models and compare performance.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion for test set
    random_state : int
        Random seed

    Returns
    -------
    dict
        Trained models
    dict
        Evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }

    # Train and evaluate
    trained_models = {}
    metrics = {}

    for name, model in models.items():
        print(f"📊 Training {name}...")

        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'Train Score': model.score(X_train, y_train),
            'Test Score': model.score(X_test, y_test)
        }

    return trained_models, metrics

def save_model(model, model_name, model_dir='models'):
    """
    Save trained model to disk.

    Parameters
    ----------
    model : object
        Trained model
    model_name : str
        Name for the model file
    model_dir : str
        Directory to save models
    """
    import os
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{model_dir}/{model_name}_{datetime.now().strftime('%Y%m%d')}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"💾 Model saved: {filename}")
    return filename

def save_metrics(metrics, filename='outputs/model_metrics.json'):
    """
    Save evaluation metrics to JSON.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics
    filename : str
        Path to save file
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj

    metrics_serializable = convert_types(metrics)

    with open(filename, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f" Metrics saved: {filename}")