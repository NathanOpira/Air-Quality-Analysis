"""
Air Quality Analysis Package

A comprehensive toolkit for analyzing air quality data with geometric insights.
"""

# Data loading and cleaning
from .data_loader import load_raw_data, clean_data, run_data_pipeline

# Feature engineering
from .features import create_temporal_features, create_pollution_features, get_feature_matrix

# Distance geometry analysis
from .distance_analyzer import GeometricAnalyzer

# Modeling
from .model_trainer import prepare_target, train_models, save_model, save_metrics

# Visualization helpers
from .visualizations import save_fig, plot_neighbor_overlap

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

__all__ = [
    # Data loading and cleaning
    'load_raw_data',
    'clean_data',
    'run_data_pipeline',

    # Feature engineering
    'create_temporal_features',
    'create_pollution_features',
    'get_feature_matrix',

    # Distance geometry analysis
    'GeometricAnalyzer',

    # Modeling
    'prepare_target',
    'train_models',
    'save_model',
    'save_metrics',

    # Visualization helpers
    'save_fig',
    'plot_neighbor_overlap'
]

# Package information
def get_info():
    """Return package information"""
    return {
        'name': 'air_quality_analysis',
        'version': __version__,
        'author': __author__,
        'modules': __all__,
        'description': 'A comprehensive toolkit for analyzing air quality data with geometric insights'
    }

# Optional: Add a convenience function to run the full pipeline
def run_full_pipeline(data_path=None, target_col='CO(GT)'):
    """
    Run the complete air quality analysis pipeline.

    Parameters:
    -----------
    data_path : str, optional
        Path to the raw data CSV file
    target_col : str
        Target column for modeling

    Returns:
    --------
    dict: Results from each stage of the pipeline
    """
    import pandas as pd
    import os

    results = {}

    # Step 1: Load and clean data
    print("Step 1: Loading and cleaning data...")
    df_raw = load_raw_data(data_path)
    df_clean, clean_meta = clean_data(df_raw)
    results['cleaned_data'] = df_clean
    results['cleaning_metadata'] = clean_meta
    print(f"  Cleaned shape: {df_clean.shape}")

    # Step 2: Feature engineering
    print("\nStep 2: Feature engineering...")
    df_temp = create_temporal_features(df_clean)
    df_feat = create_pollution_features(df_temp)
    X_features, feature_names = get_feature_matrix(df_feat)
    results['features'] = X_features
    results['feature_names'] = feature_names
    print(f"  Created {len(feature_names)} features")

    # Step 3: Distance geometry analysis
    print("\nStep 3: Distance geometry analysis...")
    # Use subset of features for analysis
    key_features = [f for f in ['CO(GT)', 'hour_sin', 'hour_cos'] if f in X_features.columns]
    if len(key_features) < 2:
        key_features = feature_names[:min(5, len(feature_names))]

    analyzer = GeometricAnalyzer(X_features, key_features)
    analyzer.compute_distance_matrices(subsample=500)
    results['analyzer'] = analyzer
    print(f"  Analyzed {len(key_features)} features with 4 distance metrics")

    # Step 4: Modeling
    print("\nStep 4: Modeling...")
    X, y = prepare_target(X_features, target_col=target_col, lag=1)
    models, metrics = train_models(X, y)
    results['models'] = models
    results['metrics'] = metrics

    # Identify best model
    best_model_name = min(metrics, key=lambda x: metrics[x]['RMSE'])
    results['best_model'] = models[best_model_name]
    results['best_model_name'] = best_model_name
    print(f"  Trained {len(models)} models, best: {best_model_name}")

    return results

# Make run_full_pipeline available if users want it
__all__.append('run_full_pipeline')