"""
Features module for feature engineering.

This module handles all feature transformation, encoding, and scaling
operations for the churn prediction pipeline.
"""

from .feature_engineering import (
    create_features,
    encode_categorical,
    scale_numerical,
    run_feature_engineering,
)

__all__ = [
    "create_features",
    "encode_categorical",
    "scale_numerical",
    "run_feature_engineering",
]
