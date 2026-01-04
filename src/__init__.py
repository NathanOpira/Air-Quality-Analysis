"""Top-level `src` package initializer.

Keep this lightweight; exposing common modules makes imports friendlier.
"""

from . import feature_engineering, distance_analysis, visualizations

__all__ = [
    'feature_engineering',
    'distance_analysis',
    'visualizations',
]

__version__ = '0.1.0'
