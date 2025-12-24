"""
Model Inference Module for RetailGenius Churn Prediction.

This module handles loading trained models and making predictions
on new data.

Usage:
    python -m src.models.inference

Author: EPITA AI PM Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib
import mlflow

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MLFLOW_EXPERIMENT_NAME,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_model(
    model_path: Optional[Path] = None, use_mlflow: bool = False
) -> object:
    """
    Load a trained model from disk or MLflow registry.

    Args:
        model_path: Path to the model file (joblib format).
        use_mlflow: Whether to load from MLflow Model Registry.

    Returns:
        Loaded model object.
    """
    if use_mlflow:
        try:
            # Load from MLflow Model Registry
            model_uri = "models:/ChurnModel/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from MLflow Registry: {model_uri}")
            return model
        except Exception as e:
            logger.warning(f"Could not load from MLflow: {e}")
            logger.info("Falling back to local model file")

    if model_path is None:
        model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run training first."
        )

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    return model


def load_preprocessors() -> tuple:
    """
    Load preprocessing objects (encoders and scaler).

    Returns:
        Tuple of (encoders dict, scaler).
    """
    encoders_path = MODELS_DIR / "encoders.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"

    encoders = joblib.load(encoders_path) if encoders_path.exists() else {}
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    logger.info("Loaded preprocessors")
    return encoders, scaler


def preprocess_new_data(
    df: pd.DataFrame, encoders: dict, scaler: object
) -> pd.DataFrame:
    """
    Preprocess new data using saved encoders and scaler.

    Args:
        df: Raw input data.
        encoders: Dictionary of label encoders.
        scaler: Fitted StandardScaler.

    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    df_processed = df.copy()

    # Apply encoders
    for col, encoder in encoders.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].apply(
                lambda x: (
                    encoder.transform([x])[0]
                    if x in encoder.classes_
                    else -1
                )
            )

    # Apply scaler to numerical columns
    if scaler is not None:
        # Get the feature names the scaler was fitted on
        feature_names_path = PROCESSED_DATA_DIR / "feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                expected_features = f.read().strip().split("\n")

            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df_processed.columns:
                    df_processed[feature] = 0

            # Reorder columns to match training data
            df_processed = df_processed[expected_features]

    return df_processed


def predict(
    model: object,
    X: Union[pd.DataFrame, np.ndarray],
    return_proba: bool = True,
) -> Union[np.ndarray, tuple]:
    """
    Make predictions using a trained model.

    Args:
        model: Trained model object.
        X: Input features.
        return_proba: Whether to return probabilities.

    Returns:
        Predictions (and probabilities if requested).
    """
    # Handle MLflow pyfunc models
    if hasattr(model, "predict"):
        if hasattr(model, "_model_impl"):
            # MLflow pyfunc wrapper
            predictions = model.predict(X)
            if return_proba and hasattr(model._model_impl, "predict_proba"):
                probabilities = model._model_impl.python_model.predict_proba(X)[:, 1]
            else:
                probabilities = None
        else:
            # Regular sklearn-like model
            predictions = model.predict(X)
            if return_proba and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = None
    else:
        raise ValueError("Model does not have a predict method")

    if return_proba and probabilities is not None:
        return predictions, probabilities
    return predictions


def run_inference(
    input_data: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run inference on input data.

    Args:
        input_data: Input DataFrame. If None, uses test data.
        output_path: Path to save predictions.

    Returns:
        DataFrame with predictions.
    """
    logger.info("=" * 50)
    logger.info("Starting Inference Pipeline")
    logger.info("=" * 50)

    # Load model
    model = load_model()

    # Load test data if no input provided
    if input_data is None:
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        logger.info(f"Using test data: {len(X_test)} samples")
    else:
        # Preprocess new data
        encoders, scaler = load_preprocessors()
        X_test = preprocess_new_data(input_data, encoders, scaler)
        logger.info(f"Preprocessed input data: {len(X_test)} samples")

    # Make predictions
    predictions, probabilities = predict(model, X_test, return_proba=True)

    # Create output DataFrame
    results = pd.DataFrame({
        "prediction": predictions,
        "churn_probability": probabilities,
        "risk_level": pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        ),
    })

    # Add original features for reference
    results = pd.concat([X_test.reset_index(drop=True), results], axis=1)

    # Save results
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "predictions.csv"

    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    # Print summary
    logger.info("\nPrediction Summary:")
    logger.info(f"  Total samples: {len(results)}")
    logger.info(f"  Predicted churners: {(predictions == 1).sum()}")
    logger.info(f"  Predicted non-churners: {(predictions == 0).sum()}")
    logger.info(f"  Average churn probability: {probabilities.mean():.2%}")
    logger.info(f"\nRisk Level Distribution:")
    logger.info(results["risk_level"].value_counts().to_string())

    logger.info("\nInference Pipeline completed successfully!")
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_inference()
