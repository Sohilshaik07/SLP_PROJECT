"""
Script: Train XLM-RoBERTa Model
================================
Fine-tunes XLM-RoBERTa on the unified multilingual
Indian code-mixed hate speech dataset.

Languages:
    - Hindi
    - Telugu
    - Tamil
    - Malayalam

Dataset:
    data/processed/unified/

Expected columns:
    text
    label
    language
    dataset

Usage
-----
    python scripts/train_xlmr.py
"""

import argparse
import sys
import logging
from pathlib import Path

import pandas as pd

# ============================================================
# Add project root to Python path
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

# ============================================================
# Project imports
# ============================================================

from src.utils.config import load_config, get_project_root
from src.models.xlmr_model import XLMRClassifier
from src.evaluation.metrics import (
    compute_metrics,
    save_metrics,
    save_predictions,
    plot_confusion_matrix
)

from src.data.loaders import DEMO_MARKER


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train XLM-RoBERTa on unified multilingual dataset"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run training on demo data"
    )

    args = parser.parse_args()

    # ========================================================
    # Load configuration
    # ========================================================

    cfg = load_config()
    project_root = get_project_root()

    # ========================================================
    # Dataset configuration
    # ========================================================

    if args.demo:

        dataset_label = "DEMO"

        train_path = (
            project_root
            / cfg["paths"]["data_processed"]
            / "hasoc2021_train.csv"
        )

        val_path = (
            project_root
            / cfg["paths"]["data_processed"]
            / "hasoc2021_val.csv"
        )

        test_path = (
            project_root
            / cfg["paths"]["data_processed"]
            / "hasoc2021_test.csv"
        )

        exp_name = "demo_indomain"

        save_subdir = "DEMO"

        is_demo = True

    else:

        dataset_label = "UNIFIED MULTILINGUAL"

        # ----------------------------------------------------
        # Unified dataset paths
        # ----------------------------------------------------

        unified_dir = (
            project_root
            / "data"
            / "processed"
            / "unified"
        )

        train_path = unified_dir / "train.csv"
        val_path = unified_dir / "val.csv"
        test_path = unified_dir / "test.csv"

        # ----------------------------------------------------
        # Experiment name
        # ----------------------------------------------------

        exp_name = "xlmr_unified_multilingual"

        # ----------------------------------------------------
        # Model save directory
        # ----------------------------------------------------

        save_subdir = "UNIFIED_XLMR_best"

        is_demo = False

    # ========================================================
    # Check dataset files
    # ========================================================

    missing_files = []

    if not train_path.exists():
        missing_files.append(str(train_path))

    if not val_path.exists():
        missing_files.append(str(val_path))

    if not test_path.exists():
        missing_files.append(str(test_path))

    if missing_files:

        print("\n" + "=" * 70)
        print("ERROR: DATASET FILES NOT FOUND")
        print("=" * 70)

        for file in missing_files:
            print(file)

        print(
            "\nPlease run:\n"
            "python scripts/prepare_unified_dataset.py\n"
        )

        sys.exit(1)

    # ========================================================
    # Load datasets
    # ========================================================

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # ========================================================
    # Dataset information
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"  TRAINING XLM-RoBERTa MODEL "
        f"({dataset_label})"
    )
    print("=" * 70)

    if is_demo:
        print(
            f"\n[WARNING] {DEMO_MARKER}\n"
        )

    print("\nDataset sizes:")
    print(f"Train: {train_df.shape}")
    print(f"Val  : {val_df.shape}")
    print(f"Test : {test_df.shape}")

    # ========================================================
    # Verify required columns
    # ========================================================

    required_columns = [
        "text",
        "label"
    ]

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"Missing required column '{column}' "
                f"in training dataset."
            )

        if column not in val_df.columns:
            raise ValueError(
                f"Missing required column '{column}' "
                f"in validation dataset."
            )

        if column not in test_df.columns:
            raise ValueError(
                f"Missing required column '{column}' "
                f"in test dataset."
            )

    # ========================================================
    # Print language distribution
    # ========================================================

    if "language" in train_df.columns:

        print("\nTraining language distribution:")
        print(
            train_df["language"]
            .value_counts()
        )

        print("\nValidation language distribution:")
        print(
            val_df["language"]
            .value_counts()
        )

        print("\nTest language distribution:")
        print(
            test_df["language"]
            .value_counts()
        )

    # ========================================================
    # Print label distribution
    # ========================================================

    print("\nTraining label distribution:")
    print(
        train_df["label"]
        .value_counts()
    )

    print("\nValidation label distribution:")
    print(
        val_df["label"]
        .value_counts()
    )

    print("\nTest label distribution:")
    print(
        test_df["label"]
        .value_counts()
    )

    # ========================================================
    # Verify missing values
    # ========================================================

    print("\nMissing values:")

    print(
        "Train:",
        train_df[["text", "label"]]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Val  :",
        val_df[["text", "label"]]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Test :",
        test_df[["text", "label"]]
        .isna()
        .sum()
        .sum()
    )

    # ========================================================
    # Initialize XLM-R model
    # ========================================================

    print("\n" + "=" * 70)
    print("INITIALIZING XLM-RoBERTa")
    print("=" * 70)

    xlmr_model = XLMRClassifier(
        config=cfg
    )

    # ========================================================
    # Model save directory
    # ========================================================

    save_dir = (
        project_root
        / "models"
        / "xlmr"
        / save_subdir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nModel directory:\n{save_dir}"
    )

    # ========================================================
    # Train XLM-RoBERTa
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING XLM-RoBERTa TRAINING")
    print("=" * 70)

    xlmr_model.train(
        train_df,
        val_df,

        # IMPORTANT:
        # Unified dataset uses:
        # text  -> input text
        # label -> target label
        text_col="text",
        label_col="label",

        save_dir=save_dir,
        demo_mode=is_demo
    )

    # ========================================================
    # Save trained model
    # ========================================================

    xlmr_model.save(
        save_dir=save_dir,
        demo_mode=is_demo
    )

    print(
        f"\nModel saved to:\n{save_dir}"
    )

    # ========================================================
    # Evaluate on TEST set
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)

    test_texts = (
        test_df["text"]
        .astype(str)
        .tolist()
    )

    y_test = (
        test_df["label"]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    preds, probs = xlmr_model.predict(
        test_texts
    )

    # ========================================================
    # Compute metrics
    # ========================================================

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=preds,
        y_proba=probs,

        # 0 = NOT
        # 1 = HATE
        label_names=[
            "NOT",
            "HATE"
        ],

        demo_mode=is_demo
    )

    # ========================================================
    # Print final metrics
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL XLM-RoBERTa TEST RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy        : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['precision_macro']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['recall_macro']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['f1_macro']:.4f}"
        "  <- PRIMARY METRIC"
    )

    print(
        f"Weighted F1     : "
        f"{metrics['f1_weighted']:.4f}"
    )

    print(
        f"AUC-ROC         : "
        f"{metrics['auc_roc']:.4f}"
    )

    # ========================================================
    # Per-class metrics
    # ========================================================

    print("\nClassification Report:")

    print(
        metrics["classification_report"]
    )

    # ========================================================
    # Save metrics
    # ========================================================

    model_key = "xlmr"

    save_metrics(
        metrics,
        model_name=model_key,
        experiment_name=exp_name,
        config=cfg,
        demo_mode=is_demo
    )

    # ========================================================
    # Save predictions
    # ========================================================

    save_predictions(
        y_true=y_test,
        y_pred=preds,
        y_proba=probs,

        model_name=model_key,
        experiment_name=exp_name,
        config=cfg,
        demo_mode=is_demo
    )

    # ========================================================
    # Save confusion matrix
    # ========================================================

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],

        label_names=[
            "NOT",
            "HATE"
        ],

        model_name=model_key,
        experiment_name=exp_name,
        config=cfg,
        demo_mode=is_demo
    )

    # ========================================================
    # Optional: Per-language evaluation
    # ========================================================
    #
    # This is VERY useful for your project because the dataset
    # is multilingual. It tells us whether XLM-R performs
    # equally well on Hindi, Telugu, Tamil and Malayalam.
    #
    # ========================================================

    if "language" in test_df.columns:

        print("\n" + "=" * 70)
        print("PER-LANGUAGE TEST PERFORMANCE")
        print("=" * 70)

        evaluation_df = test_df.copy()

        evaluation_df["prediction"] = preds

        # ----------------------------------------------------
        # Calculate per-language metrics
        # ----------------------------------------------------

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score
        )

        for language in sorted(
            evaluation_df["language"].unique()
        ):

            language_df = (
                evaluation_df[
                    evaluation_df["language"] == language
                ]
            )

            y_true_lang = (
                language_df["label"]
                .values
            )

            y_pred_lang = (
                language_df["prediction"]
                .values
            )

            accuracy = accuracy_score(
                y_true_lang,
                y_pred_lang
            )

            macro_precision = precision_score(
                y_true_lang,
                y_pred_lang,
                average="macro",
                zero_division=0
            )

            macro_recall = recall_score(
                y_true_lang,
                y_pred_lang,
                average="macro",
                zero_division=0
            )

            macro_f1 = f1_score(
                y_true_lang,
                y_pred_lang,
                average="macro",
                zero_division=0
            )

            print(
                f"\n{language}:"
            )

            print(
                f"  Samples         : "
                f"{len(language_df)}"
            )

            print(
                f"  Accuracy        : "
                f"{accuracy:.4f}"
            )

            print(
                f"  Macro Precision : "
                f"{macro_precision:.4f}"
            )

            print(
                f"  Macro Recall    : "
                f"{macro_recall:.4f}"
            )

            print(
                f"  Macro F1        : "
                f"{macro_f1:.4f}"
            )

    # ========================================================
    # Completion
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "XLM-RoBERTa MULTILINGUAL TRAINING "
        "AND EVALUATION COMPLETED"
    )
    print("=" * 70)

    print("\nGenerated outputs:")

    print(
        f"\nModel:\n{save_dir}"
    )

    print(
        "\nMetrics:"
        "\nresults/metrics/"
    )

    print(
        "\nPredictions:"
        "\nresults/predictions/"
    )

    print(
        "\nConfusion matrix:"
        "\nresults/figures/"
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()