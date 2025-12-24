"""
Configuration module for RetailGenius Churn Prediction.

This module contains all configuration variables and paths used
throughout the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = "RetailGenius_Churn_Prediction"

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "numerical_features": [
        # Standard names
        "Tenure",
        "WarehouseToHome",
        "HourSpendOnApp",
        "NumberOfDeviceRegistered",
        "SatisfactionScore",
        "NumberOfAddress",
        "Complain",
        "OrderAmountHikeFromlastYear",
        "CouponUsed",
        "OrderCount",
        "DaySinceLastOrder",
        "CashbackAmount",
        # Alternative names (spaces/variations)
        "tenure",
        "Warehouse To Home",
        "Hour Spend On App",
        "HourSpendOnApp",
        "Number Of Device Registered",
        "Satisfaction Score",
        "Number Of Address",
        "Order Amount Hike From last Year",
        "Coupon Used",
        "Order Count",
        "Day Since Last Order",
        "Cashback Amount",
    ],
    "categorical_features": [
        "PreferredLoginDevice",
        "CityTier",
        "PreferredPaymentMode",
        "Gender",
        "PreferedOrderCat",
        "MaritalStatus",
        # Alternative names
        "Preferred Login Device",
        "City Tier",
        "Preferred Payment Mode",
        "Prefered Order Cat",
        "Marital Status",
    ],
    "target": "Churn",
    "id_column": "CustomerID",
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
