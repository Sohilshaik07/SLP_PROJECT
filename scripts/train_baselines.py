"""
Train TF-IDF Baseline Models
============================

Models:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM

Dataset:
Unified multilingual dataset
Hindi + Telugu + Tamil + Malayalam
"""

import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_project_root
from src.models.baselines import TFIDFLogisticRegression, TFIDFLinearSVM
from src.evaluation.metrics import (
    compute_metrics,
    save_metrics,
    save_predictions,
    plot_confusion_matrix
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():

    # ========================================================
    # Paths
    # ========================================================

    project_root = get_project_root()

    data_dir = project_root / "data" / "processed" / "unified"

    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    # ========================================================
    # Check files
    # ========================================================

    if not train_path.exists():
        print(f"ERROR: Missing {train_path}")
        sys.exit(1)

    if not val_path.exists():
        print(f"ERROR: Missing {val_path}")
        sys.exit(1)

    if not test_path.exists():
        print(f"ERROR: Missing {test_path}")
        sys.exit(1)

    # ========================================================
    # Load datasets
    # ========================================================

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("\n" + "=" * 70)
    print("MULTILINGUAL BASELINE TRAINING")
    print("=" * 70)

    print("\nDataset sizes:")
    print("Train:", train_df.shape)
    print("Val  :", val_df.shape)
    print("Test :", test_df.shape)

    print("\nLanguages:")
    print(train_df["language"].value_counts())

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    # ========================================================
    # Prepare features and labels
    # ========================================================

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_val = val_df["text"]
    y_val = val_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    # ========================================================
    # Models
    # ========================================================

    cfg = load_config()

    models = [
        ("tfidf_lr", TFIDFLogisticRegression(config=cfg)),
        ("tfidf_svm", TFIDFLinearSVM(config=cfg)),
    ]

    # ========================================================
    # Train and evaluate
    # ========================================================

    for model_key, model_obj in models:

        print("\n" + "=" * 70)
        print(f"MODEL: {model_obj.MODEL_NAME}")
        print("=" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        print("\nTraining...")

        model_obj.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print("\nEvaluating on validation set...")

        val_preds = model_obj.predict(X_val)
        val_probs = model_obj.predict_proba(X_val)

        val_metrics = compute_metrics(
            y_true=y_val.values,
            y_pred=val_preds,
            y_proba=val_probs,
            label_names=["NOT", "HATE"]
        )

        print("\nValidation metrics:")
        print(val_metrics)

        save_metrics(
            val_metrics,
            model_name=model_key,
            experiment_name="unified_val",
            config=cfg
        )

        save_predictions(
            y_true=y_val.values,
            y_pred=val_preds,
            y_proba=val_probs,
            model_name=model_key,
            experiment_name="unified_val",
            config=cfg
        )

        plot_confusion_matrix(
            cm=val_metrics["confusion_matrix"],
            label_names=["NOT", "HATE"],
            model_name=model_key,
            experiment_name="unified_val",
            config=cfg
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        print("\nEvaluating on test set...")

        test_preds = model_obj.predict(X_test)
        test_probs = model_obj.predict_proba(X_test)

        test_metrics = compute_metrics(
            y_true=y_test.values,
            y_pred=test_preds,
            y_proba=test_probs,
            label_names=["NOT", "HATE"]
        )

        print("\nTest metrics:")
        print(test_metrics)

        save_metrics(
            test_metrics,
            model_name=model_key,
            experiment_name="unified_test",
            config=cfg
        )

        save_predictions(
            y_true=y_test.values,
            y_pred=test_preds,
            y_proba=test_probs,
            model_name=model_key,
            experiment_name="unified_test",
            config=cfg
        )

        plot_confusion_matrix(
            cm=test_metrics["confusion_matrix"],
            label_names=["NOT", "HATE"],
            model_name=model_key,
            experiment_name="unified_test",
            config=cfg
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_dir = project_root / "models" / "baselines"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"UNIFIED_{model_key}.joblib"

        model_obj.save(
            path=model_path,
            demo_mode=False
        )

        print(f"\nModel saved: {model_path}")

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()