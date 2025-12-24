"""
Model Training Module for RetailGenius Churn Prediction.

This module handles model training, evaluation, and MLflow tracking.
It trains multiple models, tracks experiments, and registers the best model.

Usage:
    python -m src.models.train

Author: EPITA AI PM Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: brew install libomp && pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

import mlflow
import mlflow.sklearn
if HAS_XGBOOST:
    import mlflow.xgboost
if HAS_LIGHTGBM:
    import mlflow.lightgbm
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_CONFIG,
    MLFLOW_EXPERIMENT_NAME,
    TRAINING_CONFIG,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load processed training and testing data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    logger.info(f"Loaded training data: {X_train.shape}")
    logger.info(f"Loaded testing data: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    return metrics


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained RandomForestClassifier.
    """
    params = MODEL_CONFIG["random_forest"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train an XGBoost classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained XGBClassifier.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed")
    params = MODEL_CONFIG["xgboost"]
    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a LightGBM classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained LGBMClassifier.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is not installed")
    params = MODEL_CONFIG["lightgbm"]
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_model(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> Any:
    """
    Train a model based on the model name.

    Args:
        model_name: Name of the model to train.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained model.
    """
    model_trainers = {
        "random_forest": train_random_forest,
    }
    
    if HAS_XGBOOST:
        model_trainers["xgboost"] = train_xgboost
    if HAS_LIGHTGBM:
        model_trainers["lightgbm"] = train_lightgbm

    if model_name not in model_trainers:
        raise ValueError(f"Unknown or unavailable model: {model_name}")

    logger.info(f"Training {model_name}...")
    model = model_trainers[model_name](X_train, y_train)
    logger.info(f"{model_name} training completed")

    return model


def run_training_pipeline() -> Dict[str, Any]:
    """
    Run the complete training pipeline with MLflow tracking.

    Returns:
        Dictionary containing best model info and metrics.
    """
    logger.info("=" * 50)
    logger.info("Starting Model Training Pipeline with MLflow")
    logger.info("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Set up MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Models to train - only include available ones
    model_names = ["random_forest"]
    if HAS_XGBOOST:
        model_names.append("xgboost")
    if HAS_LIGHTGBM:
        model_names.append("lightgbm")
    
    logger.info(f"Models to train: {model_names}")
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None

    for model_name in model_names:
        logger.info(f"\n{'='*30}")
        logger.info(f"Training: {model_name}")
        logger.info(f"{'='*30}")

        with mlflow.start_run(run_name=model_name):
            # Log parameters
            params = MODEL_CONFIG[model_name]
            mlflow.log_params(params)

            # Train model
            model = train_model(model_name, X_train, y_train)

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            # Log model
            if model_name == "random_forest":
                mlflow.sklearn.log_model(model, "model")
            elif model_name == "xgboost" and HAS_XGBOOST:
                mlflow.xgboost.log_model(model, "model")
            elif model_name == "lightgbm" and HAS_LIGHTGBM:
                mlflow.lightgbm.log_model(model, "model")

            # Save model locally
            model_path = MODELS_DIR / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            # Track results
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

            # Check if this is the best model (using F1 score)
            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_model = model
                best_model_name = model_name

    # Register best model
    logger.info(f"\nBest Model: {best_model_name} (F1 Score: {best_score:.4f})")

    # Save best model
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # Register model in MLflow Model Registry
    best_run_id = results[best_model_name]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    try:
        # Register the model
        model_details = mlflow.register_model(model_uri, "ChurnModel")
        logger.info(f"Model registered: {model_details.name} v{model_details.version}")

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="ChurnModel",
            version=model_details.version,
            stage="Production",
        )
        logger.info("Model transitioned to Production stage")
    except Exception as e:
        logger.warning(f"Could not register model: {e}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Training Summary")
    logger.info("=" * 50)
    for model_name, result in results.items():
        metrics = result["metrics"]
        logger.info(f"\n{model_name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    logger.info("\nModel Training Pipeline completed successfully!")
    logger.info("=" * 50)

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "results": results,
    }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_training_pipeline()
