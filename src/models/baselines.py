"""
Baseline Models: TF-IDF + Logistic Regression and TF-IDF + Linear SVM
=======================================================================
Both baselines use scikit-learn Pipelines combining:
  1. Feature extraction: TF-IDF (word n-grams + character n-grams)
  2. Classification: Logistic Regression / LinearSVC

IMPORTANT: These models are trained and evaluated on REAL data only.
When called with demo_mode=True, all outputs are clearly marked
"DEMO DATA — NOT PROJECT RESULTS".

Usage
-----
    from src.models.baselines import TFIDFLogisticRegression, TFIDFLinearSVM
"""

import os
import json
import logging
import joblib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)

DEMO_MARKER = "DEMO DATA — NOT PROJECT RESULTS"


def _build_tfidf_features(cfg: dict) -> FeatureUnion:
    """
    Build a FeatureUnion combining word n-grams and character n-grams.

    Parameters
    ----------
    cfg : dict
        TF-IDF config section from config.yaml.

    Returns
    -------
    FeatureUnion
    """
    word_ngram_range = tuple(cfg.get("word_ngram_range", [1, 2]))
    char_ngram_range = tuple(cfg.get("char_ngram_range", [2, 4]))
    max_features     = cfg.get("max_features", 50000)
    sublinear_tf     = cfg.get("sublinear_tf", True)
    min_df           = cfg.get("min_df", 2)

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        strip_accents=None,   # Do NOT strip accents — preserves Devanagari context
        lowercase=False,       # Preserve case (Hindi is case-insensitive, English is not)
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        strip_accents=None,
        lowercase=False,
    )

    return FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

class TFIDFLogisticRegression:
    """
    TF-IDF (word + char n-grams) → Logistic Regression classifier.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
    config : dict
    is_fitted : bool
    """

    MODEL_NAME = "TF-IDF + Logistic Regression"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.is_fitted = False
        self._build_pipeline()

    def _build_pipeline(self):
        lr_cfg   = self.config.get("logistic_regression", {})
        tfidf_cfg = lr_cfg.get("tfidf", {})
        model_cfg = lr_cfg.get("model", {})

        features = _build_tfidf_features(tfidf_cfg)
        clf = LogisticRegression(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 1000),
            solver=model_cfg.get("solver", "lbfgs"),
            random_state=model_cfg.get("random_state", 42),
            class_weight="balanced",
        )

        self.pipeline = Pipeline([
            ("features", features),
            ("classifier", clf),
        ])

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> "TFIDFLogisticRegression":
        """
        Train the pipeline.

        Parameters
        ----------
        X_train : pd.Series
            Text column (cleaned_text preferred).
        y_train : pd.Series
            Normalized integer labels (0/1).

        Returns
        -------
        self
        """
        logger.info(f"Training {self.MODEL_NAME} on {len(X_train)} samples...")
        self.pipeline.fit(X_train.astype(str), y_train)
        self.is_fitted = True
        logger.info(f"{self.MODEL_NAME} training complete.")
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(X.astype(str))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Return predicted probabilities for each class."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(X.astype(str))

    def save(self, path: Optional[str] = None, demo_mode: bool = False) -> Path:
        """
        Save the trained pipeline to disk.

        Parameters
        ----------
        path : str, optional
            Save path. Defaults to models/baselines/tfidf_lr.joblib.
        demo_mode : bool
            If True, saves to models/baselines/DEMO_tfidf_lr.joblib.

        Returns
        -------
        Path
            Path where model was saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Cannot save.")

        project_root = get_project_root()
        if path is None:
            fname = "DEMO_tfidf_lr.joblib" if demo_mode else "tfidf_lr.joblib"
            path = project_root / "models" / "baselines" / fname

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Saved {self.MODEL_NAME} → {path}")
        return path

    @classmethod
    def load(cls, path: str, config: Optional[dict] = None) -> "TFIDFLogisticRegression":
        """Load a saved model from disk."""
        instance = cls(config=config)
        instance.pipeline = joblib.load(path)
        instance.is_fitted = True
        logger.info(f"Loaded {cls.MODEL_NAME} from {path}")
        return instance


# ---------------------------------------------------------------------------
# TF-IDF + Linear SVM
# ---------------------------------------------------------------------------

class TFIDFLinearSVM:
    """
    TF-IDF (word + char n-grams) → LinearSVC classifier.

    LinearSVC does not natively produce probabilities.
    We wrap it with CalibratedClassifierCV so predict_proba() works.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
    config : dict
    is_fitted : bool
    """

    MODEL_NAME = "TF-IDF + Linear SVM"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.is_fitted = False
        self._build_pipeline()

    def _build_pipeline(self):
        svm_cfg  = self.config.get("linear_svm", {})
        tfidf_cfg = svm_cfg.get("tfidf", {})
        model_cfg = svm_cfg.get("model", {})

        features = _build_tfidf_features(tfidf_cfg)
        base_clf = LinearSVC(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 2000),
            random_state=model_cfg.get("random_state", 42),
            class_weight="balanced",
        )
        # Wrap for probability estimates
        calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)

        self.pipeline = Pipeline([
            ("features", features),
            ("classifier", calibrated_clf),
        ])

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> "TFIDFLinearSVM":
        logger.info(f"Training {self.MODEL_NAME} on {len(X_train)} samples...")
        self.pipeline.fit(X_train.astype(str), y_train)
        self.is_fitted = True
        logger.info(f"{self.MODEL_NAME} training complete.")
        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict(X.astype(str))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.pipeline.predict_proba(X.astype(str))

    def save(self, path: Optional[str] = None, demo_mode: bool = False) -> Path:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Cannot save.")

        project_root = get_project_root()
        if path is None:
            fname = "DEMO_tfidf_svm.joblib" if demo_mode else "tfidf_svm.joblib"
            path = project_root / "models" / "baselines" / fname

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Saved {self.MODEL_NAME} → {path}")
        return path

    @classmethod
    def load(cls, path: str, config: Optional[dict] = None) -> "TFIDFLinearSVM":
        instance = cls(config=config)
        instance.pipeline = joblib.load(path)
        instance.is_fitted = True
        logger.info(f"Loaded {cls.MODEL_NAME} from {path}")
        return instance
