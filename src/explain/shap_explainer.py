"""
SHAP Explainer for XLM-RoBERTa Hate Speech Classifier
=======================================================

Provides token-level SHAP explanations for the fine-tuned XLM-RoBERTa model.

Design principles
-----------------
- READ-ONLY:  Model weights are never modified.
- NO HARD-CODING: All results are computed from live model predictions.
- SELF-CONTAINED: The explainer loads the model once and reuses it.
- SAFE OUTPUTS: Results are persisted as JSON (values) and PNG (charts).

Usage
-----
    from src.explain.shap_explainer import XLMRShapExplainer

    explainer = XLMRShapExplainer()
    explainer.load_model()

    # Explain a single text
    result = explainer.explain_text("yaar teri maa ki")

    # Explain a batch and save charts
    results = explainer.explain_batch(
        texts=["text1", "text2"],
        save_dir="results/shap",
    )

    # Explain a random sample from the test set
    results = explainer.explain_test_sample(n=20, save_dir="results/shap")

Requirements
------------
    pip install shap>=0.40.0 transformers torch
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Return the project root directory."""
    try:
        from src.utils.config import get_project_root
        return get_project_root()
    except Exception:
        # Fallback: walk up from this file
        return Path(__file__).resolve().parents[2]


def _get_model_dir() -> Path:
    """Return the path to the best XLM-R checkpoint."""
    return _get_project_root() / "models" / "xlmr" / "UNIFIED_XLMR_best"


def _get_test_csv() -> Path:
    """Return the path to the unified test CSV."""
    return _get_project_root() / "data" / "processed" / "unified" / "test.csv"


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class XLMRShapExplainer:
    """
    SHAP-based explainer for the fine-tuned XLM-RoBERTa classifier.

    The explainer wraps XLMRClassifier.predict() as a SHAP-compatible
    prediction function and uses shap.maskers.Text to define the perturbation
    space.

    Attributes
    ----------
    model_dir : Path
        Directory containing the trained XLM-R checkpoint.
    classifier : XLMRClassifier or None
        Loaded classifier instance. None until load_model() is called.
    explainer : shap.Explainer or None
        SHAP explainer instance. None until load_model() is called.
    label_names : list of str
        Human-readable label names. Default ["NOT", "HOF"].
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.model_dir   = Path(model_dir) if model_dir else _get_model_dir()
        self.classifier  = None
        self.explainer   = None
        self.label_names = label_names or ["NOT", "HOF"]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> "XLMRShapExplainer":
        """
        Load the XLM-R classifier and initialise the SHAP explainer.

        This is the only step that touches the disk-loaded model.
        Model weights are NOT modified.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "shap is required. Install with: pip install shap"
            ) from e

        logger.info(f"Loading XLM-R classifier from: {self.model_dir}")
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Run the training pipeline first."
            )

        from src.models.xlmr_model import XLMRClassifier
        self.classifier = XLMRClassifier()
        self.classifier.load(str(self.model_dir))
        logger.info("XLM-R classifier loaded successfully.")

        # Build SHAP prediction function
        # Returns probability matrix of shape (n_texts, n_labels)
        def _predict_proba(texts: List[str]) -> np.ndarray:
            """Wrap XLMRClassifier.predict() as a SHAP-compatible function."""
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            _, probs = self.classifier.predict(texts)
            return np.array(probs)  # shape: (n, 2)

        # Use shap.maskers.Text for word/subword perturbation
        # The masker uses a simple whitespace tokenizer for masking positions;
        # the actual model uses its own sentencepiece tokenizer for encoding.
        masker = shap.maskers.Text(tokenizer=r"\W+")  # split on non-word chars

        logger.info("Building SHAP Explainer (this may take a moment on first call)...")
        self.explainer = shap.Explainer(
            _predict_proba,
            masker=masker,
            output_names=self.label_names,
        )
        logger.info("SHAP Explainer ready.")
        return self

    def load_from_classifier(self, classifier) -> "XLMRShapExplainer":
        """
        Build the SHAP explainer using an already-loaded XLMRClassifier.

        Use this when the classifier has already been loaded elsewhere
        (e.g. via Streamlit's @st.cache_resource) to avoid a second
        disk load of the model weights.  Model weights are NOT modified.

        Parameters
        ----------
        classifier : XLMRClassifier
            A fully loaded instance (is_fitted must be True).

        Returns
        -------
        self
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "shap is required. Install with: pip install shap"
            ) from e

        if not getattr(classifier, "is_fitted", False):
            raise RuntimeError(
                "Provided classifier is not fitted. "
                "Load the model before passing it to load_from_classifier()."
            )

        self.classifier = classifier

        def _predict_proba(texts: List[str]) -> np.ndarray:
            """SHAP-compatible probability wrapper over the loaded classifier."""
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            _, probs = self.classifier.predict(texts)
            return np.array(probs)  # shape: (n, 2)

        masker = shap.maskers.Text(tokenizer=r"\W+")
        logger.info("Building SHAP Explainer from pre-loaded classifier...")
        self.explainer = shap.Explainer(
            _predict_proba,
            masker=masker,
            output_names=self.label_names,
        )
        logger.info("SHAP Explainer ready (from pre-loaded classifier).")
        return self

    # ------------------------------------------------------------------
    # Single-text explanation
    # ------------------------------------------------------------------

    def explain_text(
        self,
        text: str,
        max_evals: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a single text.

        Parameters
        ----------
        text : str
            Input text to explain.
        max_evals : int
            Maximum number of SHAP evaluations (controls accuracy vs speed).
            Recommended: 200-1000. Higher = more accurate but slower.

        Returns
        -------
        dict with keys:
            - text (str): original input
            - tokens (list of str): words/tokens used for attribution
            - shap_values_not (list of float): per-token SHAP for NOT class
            - shap_values_hof (list of float): per-token SHAP for HOF class
            - predicted_label (str): "NOT" or "HOF"
            - predicted_prob_not (float): probability of NOT
            - predicted_prob_hof (float): probability of HOF
            - base_value_not (float): SHAP base value for NOT
            - base_value_hof (float): SHAP base value for HOF
        """
        if self.explainer is None:
            raise RuntimeError("Call load_model() before explain_text().")

        shap_values = self.explainer(
            [text],
            max_evals=max_evals,
            batch_size=1,
        )

        # shap_values.values shape: (1, n_tokens, n_classes)
        # shap_values.data[0]: token strings
        tokens  = list(shap_values.data[0])
        sv_not  = shap_values.values[0, :, 0].tolist()
        sv_hof  = shap_values.values[0, :, 1].tolist()
        base_not = float(shap_values.base_values[0, 0])
        base_hof = float(shap_values.base_values[0, 1])

        # Get model prediction
        preds, probs = self.classifier.predict([text])
        pred_label = self.label_names[int(preds[0])]
        prob_not   = float(probs[0][0])
        prob_hof   = float(probs[0][1])

        return {
            "text":               text,
            "tokens":             tokens,
            "shap_values_not":    sv_not,
            "shap_values_hof":    sv_hof,
            "predicted_label":    pred_label,
            "predicted_prob_not": prob_not,
            "predicted_prob_hof": prob_hof,
            "base_value_not":     base_not,
            "base_value_hof":     base_hof,
        }

    # ------------------------------------------------------------------
    # Batch explanation
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        texts: List[str],
        save_dir: Optional[str] = None,
        max_evals: int = 500,
        save_charts: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Compute SHAP values for a batch of texts.

        Parameters
        ----------
        texts : list of str
        save_dir : str, optional
            Directory to save JSON results and PNG charts.
            Defaults to results/shap/ in the project root.
        max_evals : int
            SHAP max evaluations per sample.
        save_charts : bool
            Whether to save bar-chart PNGs.

        Returns
        -------
        list of dict
            One result dict per input text (same format as explain_text).
        """
        if self.explainer is None:
            raise RuntimeError("Call load_model() before explain_batch().")

        if save_dir is None:
            save_dir = _get_project_root() / "results" / "shap"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(texts):
            logger.info(f"  SHAP explain [{i+1}/{len(texts)}]: {text[:60]!r}")
            result = self.explain_text(text, max_evals=max_evals)
            results.append(result)

            if save_charts:
                chart_path = save_dir / f"shap_sample_{i:04d}.png"
                self._save_bar_chart(result, chart_path)

        # Save all results as JSON
        json_path = save_dir / "shap_batch_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"SHAP batch results saved to {json_path}")

        return results

    # ------------------------------------------------------------------
    # Test-set sample explanation
    # ------------------------------------------------------------------

    def explain_test_sample(
        self,
        n: int = 20,
        save_dir: Optional[str] = None,
        max_evals: int = 500,
        random_seed: int = 42,
        text_col: str = "text",
        label_col: str = "label",
    ) -> List[Dict[str, Any]]:
        """
        Explain a random sample of n texts from the unified test set.

        Parameters
        ----------
        n : int
            Number of samples to explain.
        save_dir : str, optional
            Directory to save results (defaults to results/shap/).
        max_evals : int
            SHAP max evaluations per sample.
        random_seed : int
            Seed for reproducible sampling.
        text_col : str
            Column name for text in the test CSV.
        label_col : str
            Column name for labels in the test CSV.

        Returns
        -------
        list of dict
            SHAP results with ground-truth labels appended.
        """
        import pandas as pd

        test_csv = _get_test_csv()
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")

        df = pd.read_csv(test_csv)
        logger.info(f"Loaded test set: {len(df)} rows from {test_csv}")

        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found in test CSV. "
                f"Available: {list(df.columns)}"
            )

        # Sample n rows (stratified across labels if possible)
        rng = np.random.default_rng(random_seed)
        if label_col in df.columns:
            half = n // 2
            not_idx = df[df[label_col] == 0].index.tolist()
            hof_idx = df[df[label_col] == 1].index.tolist()
            sampled_not = rng.choice(not_idx, size=min(half, len(not_idx)), replace=False)
            sampled_hof = rng.choice(hof_idx, size=min(n - len(sampled_not), len(hof_idx)), replace=False)
            sample_idx  = np.concatenate([sampled_not, sampled_hof])
            rng.shuffle(sample_idx)
            sample_df = df.loc[sample_idx].reset_index(drop=True)
        else:
            sample_df = df.sample(n=min(n, len(df)), random_state=random_seed).reset_index(drop=True)

        logger.info(f"Explaining {len(sample_df)} sampled test texts...")
        texts = sample_df[text_col].astype(str).tolist()

        if save_dir is None:
            save_dir = _get_project_root() / "results" / "shap"
        save_dir = Path(save_dir)

        results = self.explain_batch(texts, save_dir=save_dir, max_evals=max_evals)

        # Append ground truth
        for i, result in enumerate(results):
            if label_col in sample_df.columns:
                gt_int = int(sample_df.loc[i, label_col])
                result["ground_truth_label"] = self.label_names[gt_int]
                result["ground_truth_int"]   = gt_int
            if "language" in sample_df.columns:
                result["language"] = str(sample_df.loc[i, "language"])
            if "dataset" in sample_df.columns:
                result["dataset"] = str(sample_df.loc[i, "dataset"])

        # Overwrite JSON with enriched results
        json_path = save_dir / "shap_test_sample_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"SHAP test-sample results saved to {json_path}")

        # Save summary CSV
        import csv
        summary_rows = []
        for r in results:
            summary_rows.append({
                "text":               r["text"],
                "predicted_label":    r["predicted_label"],
                "ground_truth_label": r.get("ground_truth_label", ""),
                "predicted_prob_hof": round(r["predicted_prob_hof"], 4),
                "language":           r.get("language", ""),
                "dataset":            r.get("dataset", ""),
                "top_hof_token":      _top_token(r["tokens"], r["shap_values_hof"]),
                "top_not_token":      _top_token(r["tokens"], r["shap_values_not"]),
            })

        csv_path = save_dir / "shap_test_sample_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"SHAP test-sample summary saved to {csv_path}")

        return results

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _save_bar_chart(
        self,
        result: Dict[str, Any],
        save_path: Path,
        top_k: int = 15,
        figsize: tuple = (10, 5),
    ) -> None:
        """
        Save a horizontal bar chart of per-token SHAP values for the HOF class.

        Parameters
        ----------
        result : dict
            Output from explain_text().
        save_path : Path
            File path for the PNG.
        top_k : int
            Number of top tokens (by absolute SHAP magnitude) to show.
        figsize : tuple
            Matplotlib figure size.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available -- skipping chart save.")
            return

        tokens   = result["tokens"]
        sv_hof   = np.array(result["shap_values_hof"])
        pred_lab = result["predicted_label"]
        prob_hof = result["predicted_prob_hof"]
        text_snip = result["text"][:50]

        # Select top-k by absolute value
        if len(tokens) > top_k:
            top_idx = np.argsort(np.abs(sv_hof))[::-1][:top_k]
            top_idx = sorted(top_idx)  # maintain left-to-right order
            tokens  = [tokens[i] for i in top_idx]
            sv_hof  = sv_hof[top_idx]

        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sv_hof]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, sv_hof, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value (HOF class)", fontsize=10)
        ax.set_title(
            f"SHAP Token Attribution -- Predicted: {pred_lab} "
            f"(P(HOF)={prob_hof:.3f})\n\"{text_snip}...\"",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Chart saved: {save_path}")

    def plot_text(
        self,
        text: str,
        max_evals: int = 500,
        top_k: int = 15,
        figsize: tuple = (10, 5),
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain and plot a single text, saving PNG to save_path.

        Parameters
        ----------
        text : str
            Input text.
        max_evals : int
            SHAP max evaluations.
        top_k : int
            Top tokens to display.
        figsize : tuple
            Figure size.
        save_path : str, optional
            If provided, save PNG to this path. Otherwise saves to results/shap/.

        Returns
        -------
        dict
            SHAP result dict (same as explain_text).
        """
        result = self.explain_text(text, max_evals=max_evals)

        if save_path is None:
            tmp_path = Path(_get_project_root()) / "results" / "shap" / "_tmp_plot.png"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = str(tmp_path)

        self._save_bar_chart(result, Path(save_path), top_k=top_k, figsize=figsize)
        return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _top_token(tokens: List[str], shap_values: List[float]) -> str:
    """Return the token with the highest absolute SHAP value."""
    if not tokens:
        return ""
    arr = np.abs(np.array(shap_values))
    return tokens[int(np.argmax(arr))]


def load_shap_results(json_path: str) -> List[Dict[str, Any]]:
    """
    Load previously saved SHAP results from JSON.

    Parameters
    ----------
    json_path : str
        Path to the SHAP results JSON file.

    Returns
    -------
    list of dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def shap_result_to_df(results: List[Dict[str, Any]]):
    """
    Convert a list of SHAP result dicts into a flat pandas DataFrame.

    Each row: text, predicted_label, ground_truth_label,
    predicted_prob_hof, top_hof_token, top_not_token.
    """
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "text":               r.get("text", ""),
            "predicted_label":    r.get("predicted_label", ""),
            "ground_truth_label": r.get("ground_truth_label", ""),
            "predicted_prob_hof": r.get("predicted_prob_hof", None),
            "predicted_prob_not": r.get("predicted_prob_not", None),
            "top_hof_token":      _top_token(r.get("tokens", []), r.get("shap_values_hof", [])),
            "top_not_token":      _top_token(r.get("tokens", []), r.get("shap_values_not", [])),
            "language":           r.get("language", ""),
            "dataset":            r.get("dataset", ""),
        })
    return pd.DataFrame(rows)
