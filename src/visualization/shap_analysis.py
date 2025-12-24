"""
SHAP Analysis Module for RetailGenius Churn Prediction.

This module implements Explainable AI (XAI) using SHAP
(SHapley Additive exPlanations) for model interpretability.

Usage:
    python -m src.visualization.shap_analysis

Author: EPITA AI PM Team
Date: 2025

Part 3: Explainable AI Implementation
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
)

# Configure logging
logger = logging.getLogger(__name__)


def load_model_and_data() -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the trained model and test data.

    Returns:
        Tuple of (model, X_train, X_test, y_test).
    """
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run training first."
        )
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze()

    logger.info(f"Loaded training data: {X_train.shape}")
    logger.info(f"Loaded test data: {X_test.shape}")

    return model, X_train, X_test, y_test


def create_shap_explainer(model: Any, X_train: pd.DataFrame) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for tree-based models.

    Args:
        model: Trained tree-based model.
        X_train: Training data for background.

    Returns:
        SHAP TreeExplainer object.
    """
    logger.info("Creating SHAP TreeExplainer...")

    if len(X_train) > 100:
        X_background = X_train.sample(n=100, random_state=42)
    else:
        X_background = X_train

    explainer = shap.TreeExplainer(model, X_background)
    logger.info("SHAP TreeExplainer created successfully")

    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    max_samples: Optional[int] = None,
) -> shap.Explanation:
    """
    Compute SHAP values for given data.

    Args:
        explainer: SHAP Explainer object.
        X: Data to explain.
        max_samples: Maximum number of samples.

    Returns:
        SHAP Explanation object.
    """
    if max_samples and len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    logger.info(f"Computing SHAP values for {len(X)} samples...")
    shap_values = explainer(X)
    logger.info("SHAP values computed successfully")

    return shap_values


def plot_waterfall(
    shap_values: shap.Explanation,
    sample_index: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate waterfall plot for a specific sample.

    Args:
        shap_values: SHAP Explanation object.
        sample_index: Index of the sample to explain.
        save_path: Path to save the figure.
    """
    logger.info(f"Generating waterfall plot for sample {sample_index}...")

    plt.figure(figsize=(12, 8))

    if len(shap_values.shape) == 3:
        shap.plots.waterfall(shap_values[sample_index, :, 1], show=False)
    else:
        shap.plots.waterfall(shap_values[sample_index], show=False)

    plt.title(f"SHAP Waterfall Plot - Sample {sample_index}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Waterfall plot saved to {save_path}")

    plt.close()


def plot_force(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    sample_index: int = 0,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate force plot for a specific sample.

    Args:
        explainer: SHAP Explainer object.
        shap_values: SHAP Explanation object.
        X: Feature data.
        sample_index: Index of the sample.
        save_path: Path to save the figure.
    """
    logger.info(f"Generating force plot for sample {sample_index}...")

    if hasattr(explainer, "expected_value"):
        if isinstance(explainer.expected_value, np.ndarray):
            expected_value = explainer.expected_value[1]
        else:
            expected_value = explainer.expected_value
    else:
        expected_value = shap_values.base_values[sample_index]
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]

    if len(shap_values.shape) == 3:
        sample_shap_values = shap_values.values[sample_index, :, 1]
    else:
        sample_shap_values = shap_values.values[sample_index]

    shap.force_plot(
        expected_value,
        sample_shap_values,
        X.iloc[sample_index],
        matplotlib=True,
        show=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Force plot saved to {save_path}")

    plt.close()


def plot_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_path: Optional[Path] = None,
    plot_type: str = "dot",
) -> None:
    """
    Generate summary plot showing feature importance.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature data.
        save_path: Path to save the figure.
        plot_type: Type of plot ("dot", "bar", "violin").
    """
    logger.info(f"Generating {plot_type} summary plot...")

    plt.figure(figsize=(12, 8))

    if len(shap_values.shape) == 3:
        shap.summary_plot(
            shap_values.values[:, :, 1], X, plot_type=plot_type, show=False
        )
    else:
        shap.summary_plot(shap_values.values, X, plot_type=plot_type, show=False)

    plt.title(f"SHAP Summary Plot ({plot_type.capitalize()})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Summary plot saved to {save_path}")

    plt.close()


def plot_beeswarm(
    shap_values: shap.Explanation,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate beeswarm plot showing distribution of SHAP values.

    Args:
        shap_values: SHAP Explanation object.
        save_path: Path to save the figure.
    """
    logger.info("Generating beeswarm plot...")

    plt.figure(figsize=(12, 8))

    if len(shap_values.shape) == 3:
        shap.plots.beeswarm(shap_values[:, :, 1], show=False)
    else:
        shap.plots.beeswarm(shap_values, show=False)

    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Beeswarm plot saved to {save_path}")

    plt.close()


def plot_mean_shap(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate mean absolute SHAP values bar plot.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature data.
        save_path: Path to save the figure.
    """
    logger.info("Generating mean SHAP plot...")

    plt.figure(figsize=(12, 8))

    if len(shap_values.shape) == 3:
        shap.summary_plot(shap_values.values[:, :, 1], X, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values.values, X, plot_type="bar", show=False)

    plt.title("Mean Absolute SHAP Values")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Mean SHAP plot saved to {save_path}")

    plt.close()


def plot_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Generate dependence plot for a specific feature.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature data.
        feature: Feature to plot.
        interaction_feature: Feature for interaction coloring.
        save_path: Path to save the figure.
    """
    logger.info(f"Generating dependence plot for feature: {feature}...")

    plt.figure(figsize=(10, 6))

    if len(shap_values.shape) == 3:
        values = shap_values.values[:, :, 1]
    else:
        values = shap_values.values

    shap.dependence_plot(
        feature, values, X, interaction_index=interaction_feature, show=False
    )

    plt.title(f"SHAP Dependence Plot - {feature}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Dependence plot saved to {save_path}")

    plt.close()


def plot_summary_per_class(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Generate summary plots for each class.

    Args:
        shap_values: SHAP Explanation object.
        X: Feature data.
        save_dir: Directory to save figures.
    """
    logger.info("Generating summary plots per class...")

    class_names = ["No Churn", "Churn"]

    for class_idx, class_name in enumerate(class_names):
        plt.figure(figsize=(12, 8))

        if len(shap_values.shape) == 3:
            shap.summary_plot(shap_values.values[:, :, class_idx], X, show=False)
        else:
            if class_idx == 0:
                continue
            shap.summary_plot(shap_values.values, X, show=False)

        plt.title(f"SHAP Summary Plot - Class: {class_name}")
        plt.tight_layout()

        if save_dir:
            save_path = (
                save_dir
                / f"shap_summary_class_{class_idx}_{class_name.lower().replace(' ', '_')}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot for {class_name} saved to {save_path}")

        plt.close()


def generate_shap_plots(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    sample_index: int = 0,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Generate all SHAP visualization plots.

    Args:
        explainer: SHAP Explainer object.
        shap_values: SHAP Explanation object.
        X: Feature data.
        sample_index: Index of sample for individual plots.
        output_dir: Directory to save all plots.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Generating SHAP Visualizations")
    logger.info("=" * 50)

    # 1. Waterfall plot
    plot_waterfall(
        shap_values,
        sample_index=sample_index,
        save_path=output_dir / f"shap_waterfall_sample_{sample_index}.png",
    )

    # 2. Force plot
    plot_force(
        explainer,
        shap_values,
        X,
        sample_index=sample_index,
        save_path=output_dir / f"shap_force_sample_{sample_index}.png",
    )

    # 3. Summary plot (dot)
    plot_summary(
        shap_values,
        X,
        save_path=output_dir / "shap_summary_dot.png",
        plot_type="dot",
    )

    # 4. Mean SHAP values (bar plot)
    plot_mean_shap(
        shap_values,
        X,
        save_path=output_dir / "shap_mean_values.png",
    )

    # 5. Beeswarm plot
    plot_beeswarm(
        shap_values,
        save_path=output_dir / "shap_beeswarm.png",
    )

    # 6. Summary plots per class
    plot_summary_per_class(
        shap_values,
        X,
        save_dir=output_dir,
    )

    # 7. Dependence plots for top features
    if len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    top_features_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    top_features = [X.columns[i] for i in top_features_idx]

    for feature in top_features:
        plot_dependence(
            shap_values,
            X,
            feature=feature,
            save_path=output_dir / f"shap_dependence_{feature}.png",
        )

    logger.info("All SHAP visualizations generated successfully!")


def run_shap_analysis(sample_index: int = 0, max_samples: int = 500) -> dict:
    """
    Run the complete SHAP analysis pipeline.

    Args:
        sample_index: Index of sample for individual explanations.
        max_samples: Maximum samples for SHAP computation.

    Returns:
        Dictionary containing SHAP analysis results.
    """
    logger.info("=" * 50)
    logger.info("Starting SHAP Analysis Pipeline")
    logger.info("=" * 50)

    # Load model and data
    model, X_train, X_test, y_test = load_model_and_data()

    # Create SHAP explainer
    explainer = create_shap_explainer(model, X_train)

    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X_test, max_samples=max_samples)

    # Generate all plots
    generate_shap_plots(
        explainer,
        shap_values,
        X_test.iloc[:max_samples] if max_samples else X_test,
        sample_index=sample_index,
    )

    # Create summary statistics
    if len(shap_values.shape) == 3:
        mean_abs_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
    else:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    feature_importance = pd.DataFrame(
        {"feature": X_test.columns, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info("\nTop 10 Most Important Features (by mean |SHAP|):")
    logger.info(feature_importance.head(10).to_string())

    feature_importance.to_csv(FIGURES_DIR / "shap_feature_importance.csv", index=False)
    logger.info(f"Feature importance saved to {FIGURES_DIR / 'shap_feature_importance.csv'}")

    logger.info("\nSHAP Analysis Pipeline completed successfully!")
    logger.info(f"All visualizations saved to {FIGURES_DIR}")
    logger.info("=" * 50)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_shap_analysis()
