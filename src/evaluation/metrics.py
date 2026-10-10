"""
Evaluation Metrics Module
==========================
Computes and saves evaluation metrics for all models.

Metrics computed for every experiment
--------------------------------------
- Accuracy
- Precision (macro, weighted)
- Recall (macro, weighted)
- F1 (macro, weighted)
- Full classification report
- Confusion matrix (raw counts and figure)

The primary metric for comparison is Macro F1 because it gives
equal weight to both classes, which is important for potentially
imbalanced datasets.

Usage
-----
    from src.evaluation.metrics import compute_metrics, save_metrics, plot_confusion_matrix
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)

DEMO_MARKER = "DEMO DATA — NOT PROJECT RESULTS"


# ---------------------------------------------------------------------------
# Core Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    demo_mode: bool = False,
) -> Dict:
    """
    Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0/1 integers).
    y_pred : array-like
        Predicted labels (0/1 integers).
    y_proba : array-like, optional
        Predicted probabilities of shape (n, num_classes).
        Used for AUC-ROC if provided.
    label_names : list of str, optional
        Human-readable label names for reporting.
        Default: ['NOT', 'HOF'] (HASOC convention).
    demo_mode : bool
        If True, prefixes all outputs with DEMO DATA marker.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    if label_names is None:
        label_names = ["NOT", "HOF"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    prec_per, rec_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Classification report (as string and dict)
    report_str = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # AUC-ROC (if probabilities provided)
    auc_roc = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                auc_roc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc_roc = roc_auc_score(y_true, y_proba)
            auc_roc = round(float(auc_roc), 4)
        except Exception as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")

    metrics = {
        "demo_mode":        demo_mode,
        "accuracy":         round(float(accuracy), 4),
        "precision_macro":  round(float(prec_macro), 4),
        "recall_macro":     round(float(rec_macro), 4),
        "f1_macro":         round(float(f1_macro), 4),
        "precision_weighted": round(float(prec_weighted), 4),
        "recall_weighted":  round(float(rec_weighted), 4),
        "f1_weighted":      round(float(f1_weighted), 4),
        "auc_roc":          auc_roc,
        "per_class": {
            label_names[i]: {
                "precision": round(float(prec_per[i]), 4),
                "recall":    round(float(rec_per[i]), 4),
                "f1":        round(float(f1_per[i]), 4),
                "support":   int(support_per[i]),
            }
            for i in range(len(label_names)) if i < len(prec_per)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
        "n_samples":        int(len(y_true)),
        "label_names":      label_names,
    }

    # Print summary
    marker = f"[{DEMO_MARKER}]\n" if demo_mode else ""
    print(f"\n{marker}--- Evaluation Results ---")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Macro F1        : {f1_macro:.4f}  <- primary metric")
    print(f"  Weighted F1     : {f1_weighted:.4f}")
    print(f"  Macro Precision : {prec_macro:.4f}")
    print(f"  Macro Recall    : {rec_macro:.4f}")
    if auc_roc is not None:
        print(f"  AUC-ROC         : {auc_roc:.4f}")
    print(f"\n  Classification Report:\n{report_str}")

    return metrics


# ---------------------------------------------------------------------------
# Save / Load Metrics
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: Dict,
    model_name: str,
    experiment_name: str,
    config: Optional[dict] = None,
    demo_mode: bool = False,
) -> Path:
    """
    Save metrics dictionary to results/metrics/.

    Parameters
    ----------
    metrics : dict
        Output from compute_metrics().
    model_name : str
        Short model identifier (e.g., 'tfidf_lr', 'xlmr').
    experiment_name : str
        Experiment identifier (e.g., 'hasoc2021_indomain').
    config : dict, optional
    demo_mode : bool

    Returns
    -------
    Path
        Path to saved JSON file.
    """
    cfg = config or load_config()
    project_root = get_project_root()
    metrics_dir = project_root / cfg["paths"]["results_metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prefix = "DEMO_" if demo_mode else ""
    fname = f"{prefix}{model_name}_{experiment_name}_metrics.json"
    out_path = metrics_dir / fname

    metrics["model_name"]      = model_name
    metrics["experiment_name"] = experiment_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved → {out_path}")
    return out_path


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    experiment_name: str,
    config: Optional[dict] = None,
    demo_mode: bool = False,
) -> Path:
    """Save predictions CSV to results/predictions/."""
    cfg = config or load_config()
    project_root = get_project_root()
    pred_dir = project_root / cfg["paths"]["results_predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    prefix = "DEMO_" if demo_mode else ""
    fname = f"{prefix}{model_name}_{experiment_name}_predictions.csv"
    out_path = pred_dir / fname

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if y_proba is not None:
        if y_proba.ndim == 2:
            df["prob_NOT"] = y_proba[:, 0]
            df["prob_HOF"] = y_proba[:, 1]
        else:
            df["prob_positive"] = y_proba

    if demo_mode:
        df["demo_marker"] = DEMO_MARKER

    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Predictions saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Confusion Matrix Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    model_name: str,
    experiment_name: str,
    config: Optional[dict] = None,
    demo_mode: bool = False,
) -> Path:
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.metrics.confusion_matrix.
    label_names : list of str
    model_name : str
    experiment_name : str
    config : dict, optional
    demo_mode : bool

    Returns
    -------
    Path
        Path to saved figure.
    """
    cfg = config or load_config()
    project_root = get_project_root()
    fig_dir = project_root / cfg["paths"]["results_figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    prefix = "DEMO_" if demo_mode else ""
    fname = f"{prefix}{model_name}_{experiment_name}_confusion_matrix.png"
    out_path = fig_dir / fname

    cm_array = np.array(cm)
    cm_norm  = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(
        cm_array, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion Matrix (Counts)\n{model_name}")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1],
    )
    axes[1].set_title(f"Confusion Matrix (Normalized)\n{model_name}")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    title_suffix = f"\n[WARNING] {DEMO_MARKER}" if demo_mode else ""
    fig.suptitle(
        f"Experiment: {experiment_name}{title_suffix}",
        fontsize=11, fontweight="bold", color="red" if demo_mode else "black",
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Model Comparison Table
# ---------------------------------------------------------------------------

def build_comparison_table(
    results_dir: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load all saved metrics JSON files and build a comparison table.

    Parameters
    ----------
    results_dir : str, optional
        Directory containing *_metrics.json files.
    config : dict, optional

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model/experiment.
    """
    cfg = config or load_config()
    project_root = get_project_root()

    if results_dir is None:
        results_dir = project_root / cfg["paths"]["results_metrics"]
    else:
        results_dir = Path(results_dir)

    metric_files = sorted(results_dir.glob("*_metrics.json"))

    if not metric_files:
        logger.warning(f"No metric files found in {results_dir}")
        return pd.DataFrame()

    rows = []
    for fpath in metric_files:
        with open(fpath, encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "Model":          m.get("model_name", "unknown"),
            "Experiment":     m.get("experiment_name", "unknown"),
            "Demo Mode":      m.get("demo_mode", False),
            "N Samples":      m.get("n_samples", "-"),
            "Accuracy":       m.get("accuracy", "-"),
            "Precision (M)":  m.get("precision_macro", "-"),
            "Recall (M)":     m.get("recall_macro", "-"),
            "F1 Macro":       m.get("f1_macro", "-"),
            "F1 Weighted":    m.get("f1_weighted", "-"),
            "AUC-ROC":        m.get("auc_roc", "-"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
