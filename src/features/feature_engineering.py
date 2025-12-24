"""
Feature Engineering Module for RetailGenius Churn Prediction.

This module handles all feature transformation, encoding, scaling,
and feature creation operations.

Usage:
    python -m src.features.feature_engineering

Author: EPITA AI PM Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FEATURE_CONFIG,
    LOGGING_CONFIG,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_cleaned_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load cleaned data from interim directory.

    Args:
        filepath: Path to the cleaned data file.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    if filepath is None:
        filepath = INTERIM_DATA_DIR / "cleaned_data.csv"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {filepath}. "
            "Please run data preparation first."
        )

    logger.info(f"Loading cleaned data from {filepath}")
    return pd.read_csv(filepath)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.

    Args:
        df: DataFrame with cleaned data.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    logger.info("Creating derived features...")
    df_features = df.copy()

    # Feature: Average order value (if we have order amount and count)
    if "CashbackAmount" in df_features.columns and "OrderCount" in df_features.columns:
        df_features["AvgCashbackPerOrder"] = (
            df_features["CashbackAmount"] / df_features["OrderCount"].replace(0, 1)
        )
        logger.info("Created feature: AvgCashbackPerOrder")

    # Feature: Engagement score
    if "HourSpendOnApp" in df_features.columns and "NumberOfDeviceRegistered" in df_features.columns:
        df_features["EngagementScore"] = (
            df_features["HourSpendOnApp"] * df_features["NumberOfDeviceRegistered"]
        )
        logger.info("Created feature: EngagementScore")

    # Feature: Distance impact
    if "WarehouseToHome" in df_features.columns and "Tenure" in df_features.columns:
        df_features["DistancePerTenure"] = (
            df_features["WarehouseToHome"] / df_features["Tenure"].replace(0, 1)
        )
        logger.info("Created feature: DistancePerTenure")

    # Feature: Customer activity level
    if "DaySinceLastOrder" in df_features.columns and "OrderCount" in df_features.columns:
        # Higher score = more active
        df_features["ActivityLevel"] = (
            df_features["OrderCount"] / df_features["DaySinceLastOrder"].replace(0, 1)
        )
        logger.info("Created feature: ActivityLevel")

    # Feature: Coupon usage rate
    if "CouponUsed" in df_features.columns and "OrderCount" in df_features.columns:
        df_features["CouponUsageRate"] = (
            df_features["CouponUsed"] / df_features["OrderCount"].replace(0, 1)
        )
        logger.info("Created feature: CouponUsageRate")

    # Feature: Complaint flag (binary)
    if "Complain" in df_features.columns:
        df_features["HasComplained"] = (df_features["Complain"] > 0).astype(int)
        logger.info("Created feature: HasComplained")

    # Feature: Tenure category
    if "Tenure" in df_features.columns:
        df_features["TenureCategory"] = pd.cut(
            df_features["Tenure"],
            bins=[0, 6, 12, 24, float("inf")],
            labels=["New", "Growing", "Established", "Loyal"],
        )
        logger.info("Created feature: TenureCategory")

    logger.info(f"Feature creation completed. Total columns: {len(df_features.columns)}")
    return df_features


def encode_categorical(
    df: pd.DataFrame, fit: bool = True, encoders: Optional[dict] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables using Label Encoding.

    Args:
        df: DataFrame with categorical columns.
        fit: Whether to fit new encoders or use existing.
        encoders: Dictionary of fitted encoders (if fit=False).

    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders).
    """
    logger.info("Encoding categorical variables...")
    df_encoded = df.copy()

    if encoders is None:
        encoders = {}

    categorical_cols = [
        col
        for col in FEATURE_CONFIG["categorical_features"]
        if col in df_encoded.columns
    ]

    # Also include derived categorical features
    if "TenureCategory" in df_encoded.columns:
        categorical_cols.append("TenureCategory")

    for col in categorical_cols:
        if fit:
            encoder = LabelEncoder()
            # Handle unseen labels by converting to string first
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = encoder.fit_transform(df_encoded[col])
            encoders[col] = encoder
            logger.info(f"Encoded '{col}' with {len(encoder.classes_)} classes")
        else:
            if col in encoders:
                df_encoded[col] = df_encoded[col].astype(str)
                # Handle unseen labels
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: (
                        encoders[col].transform([x])[0]
                        if x in encoders[col].classes_
                        else -1
                    )
                )

    return df_encoded, encoders


def scale_numerical(
    df: pd.DataFrame, fit: bool = True, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        df: DataFrame with numerical columns.
        fit: Whether to fit a new scaler or use existing.
        scaler: Fitted StandardScaler (if fit=False).

    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    logger.info("Scaling numerical features...")
    df_scaled = df.copy()

    # Get numerical columns that exist in the dataframe
    numerical_cols = [
        col
        for col in FEATURE_CONFIG["numerical_features"]
        if col in df_scaled.columns
    ]

    # Also include derived numerical features
    derived_numerical = [
        "AvgCashbackPerOrder",
        "EngagementScore",
        "DistancePerTenure",
        "ActivityLevel",
        "CouponUsageRate",
    ]
    numerical_cols.extend([col for col in derived_numerical if col in df_scaled.columns])

    # If no predefined columns found, use all numeric columns except target
    if not numerical_cols:
        logger.warning("No predefined numerical columns found. Using all numeric columns.")
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target column if present
        target_col = FEATURE_CONFIG.get("target", "Churn")
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)

    logger.info(f"Numerical columns to scale: {numerical_cols}")

    if not numerical_cols:
        logger.warning("No numerical columns to scale. Returning original dataframe.")
        return df_scaled, StandardScaler()

    if fit:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        logger.info(f"Fitted scaler on {len(numerical_cols)} numerical columns")
    else:
        if scaler is not None:
            df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

    return df_scaled, scaler


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df: Full dataset.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    target_col = FEATURE_CONFIG["target"]
    id_col = FEATURE_CONFIG["id_column"]

    # Separate features and target
    feature_cols = [
        col for col in df.columns if col not in [target_col, id_col]
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def save_processed_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoders: dict,
    scaler: StandardScaler,
) -> None:
    """
    Save processed features and preprocessing objects.

    Args:
        X_train, X_test: Feature DataFrames.
        y_train, y_test: Target Series.
        encoders: Dictionary of label encoders.
        scaler: Fitted StandardScaler.
    """
    # Save datasets
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    # Save preprocessors
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # Save feature names
    with open(PROCESSED_DATA_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(X_train.columns.tolist()))

    logger.info(f"Saved processed data to {PROCESSED_DATA_DIR}")
    logger.info(f"Saved preprocessors to {MODELS_DIR}")


def run_feature_engineering() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the complete feature engineering pipeline.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info("=" * 50)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 50)

    # Step 1: Load cleaned data
    df = load_cleaned_data()

    # Step 2: Create derived features
    df = create_features(df)

    # Step 3: Encode categorical variables
    df, encoders = encode_categorical(df, fit=True)

    # Step 4: Scale numerical features
    df, scaler = scale_numerical(df, fit=True)

    # Step 5: Split into train/test
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    # Step 6: Save everything
    save_processed_features(X_train, X_test, y_train, y_test, encoders, scaler)

    logger.info("Feature Engineering Pipeline completed successfully!")
    logger.info("=" * 50)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_feature_engineering()
