"""
Visualization module for plotting and explainability.

This module handles all visualization tasks including
standard plots and SHAP-based model explanations.
"""

from .plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
)
from .shap_analysis import (
    create_shap_explainer,
    generate_shap_plots,
    run_shap_analysis,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "create_shap_explainer",
    "generate_shap_plots",
    "run_shap_analysis",
]
