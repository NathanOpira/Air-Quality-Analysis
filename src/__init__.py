"""
Air Quality Analysis Package
"""
from .data_loader import load_raw_data, clean_data, run_data_pipeline
from .features import create_temporal_features, create_pollution_features, get_feature_matrix
from .distance_analyzer import GeometricAnalyzer
from .model_trainer import prepare_target, train_models, save_model, save_metrics

__version__ = '1.0.0'
__all__ = [
    'load_raw_data',
    'clean_data',
    'run_data_pipeline',
    'create_temporal_features',
    'create_pollution_features',
    'get_feature_matrix',
    'GeometricAnalyzer',
    'prepare_target',
    'train_models',
    'save_model',
    'save_metrics'
]