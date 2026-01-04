import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def prepare_target(data, target_col='CO(GT)', lag=1):
    """Prepare data for time series prediction"""
    if target_col not in data.columns:
        # Try to find alternative target
        pollutant_patterns = ['CO', 'NO', 'NOx', 'NO2', 'O3', 'TEMP', 'RH', 'PM']
        for pattern in pollutant_patterns:
            for col in data.columns:
                if pattern in col.upper() and data[col].dtype in [np.float64, np.int64]:
                    target_col = col
                    break
            if target_col in data.columns:
                break

        if target_col not in data.columns:
            target_col = data.columns[-1]

    # Create lagged target
    y = data[target_col].shift(-lag).dropna()
    X = data.iloc[:len(y)].copy()

    # Ensure same length
    X = X.iloc[:len(y)]
    y = y.iloc[:len(X)]

    return X, y


def train_models(X, y, test_size=0.2, random_state=42):
    """Train and evaluate multiple regression models"""
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    }

    # Train and evaluate
    models = {}
    metrics = {}

    for name, model in models_dict.items():
        model.fit(X_train_scaled, y_train)
        models[name] = model

        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate metrics
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

        metrics[name] = {
            'Train Score': train_score,
            'Test Score': test_score,
            'MAE': mae,
            'RMSE': rmse,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        }

    return models, metrics


def save_model(model, model_name, save_dir='../models'):
    """Save trained model to disk"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    return filepath


def save_metrics(metrics, save_dir='../outputs/metrics'):
    """Save metrics to CSV"""
    os.makedirs(save_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics).T
    filepath = os.path.join(save_dir, 'model_metrics.csv')
    metrics_df.to_csv(filepath)
    return filepath