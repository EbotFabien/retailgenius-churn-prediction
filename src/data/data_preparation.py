"""
Data Preparation Module for RetailGenius Churn Prediction.

This module handles all data loading, cleaning, and validation operations.
It reads raw data, handles missing values, removes duplicates, and saves
the cleaned data for feature engineering.

Usage:
    python -m src.data.data_preparation

Author: EPITA AI PM Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    FEATURE_CONFIG,
    LOGGING_CONFIG,
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG) if hasattr(logging, 'config') else None
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw data from CSV or Excel file.

    Args:
        filepath: Path to the data file. If None, uses default location.

    Returns:
        pd.DataFrame: Raw data loaded from file.

    Raises:
        FileNotFoundError: If the data file is not found.
    """
    if filepath is None:
        # Try common file names
        possible_names = [
            "E Commerce Dataset.xlsx",
            "E_Commerce_Dataset.xlsx",
            "ecommerce_churn.csv",
            "churn_data.csv",
            "data.csv",
            "E-commerce Customer Churn.xlsx",
        ]
        for name in possible_names:
            filepath = RAW_DATA_DIR / name
            if filepath.exists():
                break
        else:
            # List available files
            available = list(RAW_DATA_DIR.glob("*"))
            raise FileNotFoundError(
                f"No data file found in {RAW_DATA_DIR}. "
                f"Available files: {available}"
            )

    logger.info(f"Loading data from {filepath}")

    # Load based on file extension
    if filepath.suffix == ".xlsx" or filepath.suffix == ".xls":
        # Try to find the correct sheet with data
        xl = pd.ExcelFile(filepath)
        logger.info(f"Excel sheets found: {xl.sheet_names}")
        
        df = None
        for sheet_name in xl.sheet_names:
            temp_df = pd.read_excel(xl, sheet_name=sheet_name)
            # Check if this sheet has meaningful data (more than 4 columns and has 'Churn' or customer data)
            if len(temp_df.columns) > 4:
                # Check for common column names
                cols_lower = [str(c).lower() for c in temp_df.columns]
                if any(keyword in ' '.join(cols_lower) for keyword in ['churn', 'customer', 'tenure', 'order']):
                    df = temp_df
                    logger.info(f"Using sheet: {sheet_name}")
                    break
        
        # If no sheet found with keywords, use the one with most columns
        if df is None:
            best_sheet = None
            max_cols = 0
            for sheet_name in xl.sheet_names:
                temp_df = pd.read_excel(xl, sheet_name=sheet_name)
                if len(temp_df.columns) > max_cols:
                    max_cols = len(temp_df.columns)
                    best_sheet = sheet_name
                    df = temp_df
            logger.info(f"Using sheet with most columns: {best_sheet} ({max_cols} columns)")
        
        if df is None:
            df = pd.read_excel(filepath, sheet_name=0)
    else:
        df = pd.read_csv(filepath)

    # Clean column names - remove leading/trailing spaces
    df.columns = df.columns.str.strip() if hasattr(df.columns, 'str') else df.columns
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate the loaded data for required columns and data types.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple of (is_valid, list of issues found).
    """
    issues = []

    # Check for required columns
    required_cols = (
        FEATURE_CONFIG["numerical_features"]
        + FEATURE_CONFIG["categorical_features"]
        + [FEATURE_CONFIG["target"]]
    )

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")

    # Check target variable
    if FEATURE_CONFIG["target"] in df.columns:
        unique_targets = df[FEATURE_CONFIG["target"]].nunique()
        if unique_targets < 2:
            issues.append(f"Target variable has only {unique_targets} unique value(s)")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Data validation passed")
    else:
        logger.warning(f"Data validation issues: {issues}")

    return is_valid, issues


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data by handling missing values and duplicates.

    Args:
        df: Raw DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Starting data cleaning...")
    initial_rows = len(df)

    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")

    # Handle missing values in numerical columns
    numerical_cols = [
        col for col in FEATURE_CONFIG["numerical_features"] if col in df_clean.columns
    ]
    for col in numerical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            # Use median for numerical columns
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
            logger.info(
                f"Filled {missing_count} missing values in '{col}' with median: {median_value}"
            )

    # Handle missing values in categorical columns
    categorical_cols = [
        col
        for col in FEATURE_CONFIG["categorical_features"]
        if col in df_clean.columns
    ]
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            # Use mode for categorical columns
            mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
            df_clean[col] = df_clean[col].fillna(mode_value)
            logger.info(
                f"Filled {missing_count} missing values in '{col}' with mode: {mode_value}"
            )

    # Handle outliers using IQR method for numerical columns
    for col in numerical_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            # Cap outliers instead of removing
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped {outliers} outliers in '{col}'")

    final_rows = len(df_clean)
    logger.info(
        f"Data cleaning completed. Rows: {initial_rows} -> {final_rows}"
    )

    return df_clean


def save_processed_data(df: pd.DataFrame, filename: str = "cleaned_data.csv") -> Path:
    """
    Save cleaned data to the interim directory.

    Args:
        df: Cleaned DataFrame to save.
        filename: Name of the output file.

    Returns:
        Path to the saved file.
    """
    output_path = INTERIM_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    return output_path


def generate_data_report(df: pd.DataFrame) -> dict:
    """
    Generate a summary report of the data.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dictionary containing data statistics.
    """
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }

    # Add target distribution if present
    if FEATURE_CONFIG["target"] in df.columns:
        report["target_distribution"] = df[FEATURE_CONFIG["target"]].value_counts().to_dict()

    return report


def run_data_preparation() -> pd.DataFrame:
    """
    Run the complete data preparation pipeline.

    Returns:
        pd.DataFrame: Cleaned and validated data.
    """
    logger.info("=" * 50)
    logger.info("Starting Data Preparation Pipeline")
    logger.info("=" * 50)

    # Step 1: Load data
    df = load_raw_data()

    # Step 2: Validate data
    is_valid, issues = validate_data(df)
    if not is_valid:
        logger.warning(f"Validation issues found: {issues}")
        # Continue anyway, but log warnings

    # Step 3: Clean data
    df_clean = clean_data(df)

    # Step 4: Generate report
    report = generate_data_report(df_clean)
    logger.info(f"Data Report: {report['n_rows']} rows, {report['n_columns']} columns")

    # Step 5: Save cleaned data
    save_processed_data(df_clean)

    logger.info("Data Preparation Pipeline completed successfully!")
    logger.info("=" * 50)

    return df_clean


if __name__ == "__main__":
    # Run the data preparation pipeline
    run_data_preparation()
