"""
Plotting Module for RetailGenius Churn Prediction.

This module provides standard visualization functions for
model evaluation and data analysis.

Author: EPITA AI PM Team
Date: 2025
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FIGURES_DIR

# Configure logging
logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels.
        save_path: Path to save the figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if labels is None:
        labels = ["No Churn", "Churn"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        save_path: Path to save the figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15,
    save_path: Optional[Path] = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """
    Plot feature importance from a tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to display.
        save_path: Path to save the figure.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return None

    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def plot_churn_distribution(
    df: pd.DataFrame,
    target_col: str = "Churn",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the distribution of churn in the dataset.

    Args:
        df: DataFrame with target column.
        target_col: Name of the target column.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    sns.countplot(data=df, x=target_col, ax=axes[0], palette="Set2")
    axes[0].set_title("Churn Distribution (Count)")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")

    # Pie chart
    churn_counts = df[target_col].value_counts()
    axes[1].pie(
        churn_counts.values,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff9999"],
    )
    axes[1].set_title("Churn Distribution (Percentage)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Churn distribution plot saved to {save_path}")

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot correlation matrix for numerical features.

    Args:
        df: DataFrame with numerical columns.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])

    corr_matrix = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Correlation matrix saved to {save_path}")

    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target_col: str = "Churn",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot distributions of numerical features by churn status.

    Args:
        df: DataFrame with features.
        numerical_cols: List of numerical column names.
        target_col: Target column name.
        save_path: Path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_cols):
        if col in df.columns:
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[idx], palette="Set2")
            axes[idx].set_title(f"{col} by Churn Status")

    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Numerical distributions saved to {save_path}")

    return fig
