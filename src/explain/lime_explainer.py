"""
LIME Explainer for XLM-RoBERTa Hate Speech Classifier
=====================================================

Provides token-level / word-level LIME (Local Interpretable Model-agnostic
Explanations) for the fine-tuned XLM-RoBERTa model on Indian code-mixed text.

Scientific Design Principles
----------------------------
1. READ-ONLY: Model weights are never modified.
2. NO HARD-CODING: All explanations are generated from live model predictions.
3. INDEPENDENCE: LIME is completely independent of SHAP; no values or artifacts
   are shared or copied between methods.
4. LOCAL SURROGATE: LIME trains an interpretable local surrogate linear model
   on perturbed text samples. Weights represent local feature sensitivity,
   NOT causal importance.
5. SUBWORD RELATION: LIME perturbs input texts at word/whitespace boundaries.
   XLM-RoBERTa subsequently processes each perturbed text with its native
   SentencePiece subword tokenizer.
6. CACHING SUPPORT: Supports loading from an existing XLMRClassifier instance
   (e.g., Streamlit @st.cache_resource) to prevent redundant disk I/O.

Usage
-----
    from src.explain.lime_explainer import XLMRLimeExplainer

    explainer = XLMRLimeExplainer()
    explainer.load_model("models/xlmr/UNIFIED_XLMR_best")

    # Or reuse existing cached classifier:
    # explainer.load_from_classifier(cached_classifier)

    result = explainer.explain_text("ye saale sab chor hain", num_samples=250)
    print(result["predicted_class_name"])
    for feat, wt, direction in zip(result["features"], result["weights"], result["directions"]):
        print(f"  {feat}: {wt:+.4f} {direction}")

Requirements
------------
    pip install lime>=0.2.0 transformers torch
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Return the project root directory."""
    try:
        from src.utils.config import get_project_root
        return get_project_root()
    except Exception:
        return Path(__file__).resolve().parents[2]


class XLMRLimeExplainer:
    """
    Local Interpretable Model-agnostic Explainer (LIME) for XLMRClassifier.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to fine-tuned XLM-RoBERTa directory.
    config : dict, optional
        Project configuration dictionary.
    label_names : list of str, optional
        Human-readable class names. Default ["Non-Hate", "Hate"].
    random_state : int, optional
        Random seed for perturbation reproducibility. Default 42.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
        label_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.config = config
        self.label_names = label_names or ["Non-Hate", "Hate"]
        self.random_state = random_state

        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            root = _get_project_root()
            self.model_path = root / "models" / "xlmr" / "UNIFIED_XLMR_best"

        self.classifier = None
        self.explainer = None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> "XLMRLimeExplainer":
        """
        Load the XLMRClassifier from disk and initialize LIME explainer.

        Model weights are loaded in read-only evaluation mode.
        """
        if model_path is not None:
            self.model_path = Path(model_path)

        logger.info(f"Loading XLMRClassifier from: {self.model_path}")
        from src.models.xlmr_model import XLMRClassifier

        self.classifier = XLMRClassifier(config=self.config)
        self.classifier.load(str(self.model_path))

        self._build_lime_explainer()
        return self

    def load_from_classifier(self, classifier) -> "XLMRLimeExplainer":
        """
        Initialize LIME explainer reusing an already-loaded XLMRClassifier.

        Avoids reloading model weights into memory when already cached
        (e.g., in Streamlit). Model weights are NOT modified.

        Parameters
        ----------
        classifier : XLMRClassifier
            A loaded, fitted instance.

        Returns
        -------
        self
        """
        if classifier is None:
            raise ValueError("Provided classifier cannot be None.")
        if not getattr(classifier, "is_fitted", False):
            raise RuntimeError("Classifier must be fitted / loaded before building LIME explainer.")

        self.classifier = classifier
        self._build_lime_explainer()
        return self

    def _build_lime_explainer(self) -> None:
        """Instantiate the LimeTextExplainer instance."""
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as exc:
            raise ImportError(
                "lime package is required for XLMRLimeExplainer. "
                "Install with: pip install lime>=0.2.0"
            ) from exc

        logger.info("Initializing LimeTextExplainer...")
        self.explainer = LimeTextExplainer(
            class_names=self.label_names,
            random_state=self.random_state,
            bow=False,  # Preserve word occurrences rather than collapsing duplicates
            split_expression=r"\W+",  # Robust word boundary split for multilingual / code-mixed text
        )
        logger.info("LIME Text Explainer ready.")

    # ------------------------------------------------------------------
    # Prediction Function for LIME
    # ------------------------------------------------------------------

    def _predict_proba_fn(self, texts: List[str]) -> np.ndarray:
        """
        Black-box prediction interface for LIME perturbations.

        Parameters
        ----------
        texts : list of str
            Perturbed text samples generated by LIME.

        Returns
        -------
        np.ndarray of shape (len(texts), 2)
            Probability distribution [P(Non-Hate), P(Hate)] for each perturbation.
            Guaranteed: P(Non-Hate) + P(Hate) == 1.0 within float precision.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier is not loaded. Call load_model() or load_from_classifier() first.")

        # Clean/sanitize texts: handle empty strings or whitespace-only perturbations
        clean_texts = [str(t) if str(t).strip() else " " for t in texts]

        # Use actual model's predict method
        _, probs = self.classifier.predict(clean_texts)
        probs = np.asarray(probs, dtype=np.float64)

        # Safety check: normalize rows if needed so sum = 1.0 exactly
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized_probs = probs / row_sums

        return normalized_probs

    # ------------------------------------------------------------------
    # Single-Text Explanation
    # ------------------------------------------------------------------

    def explain_text(
        self,
        text: str,
        target_class: int = 1,
        num_features: int = 10,
        num_samples: int = 300,
    ) -> Dict[str, Any]:
        """
        Generate a LIME explanation for a single input text.

        Parameters
        ----------
        text : str
            The input sentence (multilingual or code-mixed).
        target_class : int, optional
            The class to explain. 1 for Hate (default), 0 for Non-Hate.
        num_features : int, optional
            Number of top contributing features to return. Default 10.
        num_samples : int, optional
            Number of perturbation samples evaluated by LIME. Default 300.

        Returns
        -------
        dict with structured explanation data:
            - input_text: original input
            - predicted_label: int (0 or 1)
            - predicted_class_name: "Non-Hate" or "Hate"
            - probability_non_hate: float
            - probability_hate: float
            - confidence: float
            - explained_class: int
            - explained_class_name: str
            - features: list of str
            - weights: list of float
            - directions: list of str ("-> Hate" or "-> Non-Hate")
            - raw_explanation: list of (str, float) tuples
            - score: float (local linear surrogate model R^2 fit score)
            - num_samples: int
            - method: "LIME"
        """
        if not text or not str(text).strip():
            raise ValueError("Input text cannot be empty.")

        if self.explainer is None:
            raise RuntimeError("LIME explainer not initialized. Call load_model() or load_from_classifier().")

        clean_text = str(text).strip()

        # 1. Live model prediction on original unperturbed input
        preds, probs = self.classifier.predict([clean_text])
        pred_label = int(preds[0])
        prob_non_hate = float(probs[0][0])
        prob_hate = float(probs[0][1])
        confidence = float(probs[0][pred_label])
        pred_class_name = self.label_names[pred_label]

        # 2. Run LIME local surrogate fitting
        exp = self.explainer.explain_instance(
            text_instance=clean_text,
            classifier_fn=self._predict_proba_fn,
            labels=(target_class,),
            num_features=num_features,
            num_samples=num_samples,
        )

        raw_list = exp.as_list(label=target_class)
        explained_class_name = self.label_names[target_class]

        features = []
        weights = []
        directions = []

        for feat, wt in raw_list:
            feat_str = str(feat).strip()
            wt_float = float(wt)
            features.append(feat_str)
            weights.append(wt_float)

            # Determine direction relative to target class
            if target_class == 1:
                # Target is Hate:
                # Positive weight pushes toward Hate
                # Negative weight pushes toward Non-Hate
                direction = "-> Hate" if wt_float > 0 else "-> Non-Hate"
            else:
                # Target is Non-Hate:
                # Positive weight pushes toward Non-Hate
                # Negative weight pushes toward Hate
                direction = "-> Non-Hate" if wt_float > 0 else "-> Hate"
            directions.append(direction)

        # Local surrogate score (R^2 score of ridge regressor)
        surrogate_score = float(getattr(exp, "score", 0.0) or 0.0)

        return {
            "method": "LIME",
            "input_text": clean_text,
            "predicted_label": pred_label,
            "predicted_class_name": pred_class_name,
            "probability_non_hate": prob_non_hate,
            "probability_hate": prob_hate,
            "confidence": confidence,
            "explained_class": target_class,
            "explained_class_name": explained_class_name,
            "features": features,
            "weights": weights,
            "directions": directions,
            "raw_explanation": [(str(f), float(w)) for f, w in raw_list],
            "score": surrogate_score,
            "num_samples": num_samples,
        }

    # ------------------------------------------------------------------
    # Batch Explanation & Evaluation
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        texts: List[str],
        target_class: int = 1,
        num_features: int = 10,
        num_samples: int = 250,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Explain a batch of texts and optionally save results and charts.

        Parameters
        ----------
        texts : list of str
            Input texts.
        target_class : int
            Target class index to explain (default 1 for Hate).
        num_features : int
            Number of features to return per text.
        num_samples : int
            Perturbation samples per explanation.
        save_dir : str or Path, optional
            Directory to save JSON artifacts and PNG figures.

        Returns
        -------
        list of dict
            Explanation dictionaries.
        """
        results = []
        out_dir = Path(save_dir) if save_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(texts):
            logger.info(f"Generating LIME explanation {i+1}/{len(texts)}: {text[:40]}...")
            exp_res = self.explain_text(
                text=text,
                target_class=target_class,
                num_features=num_features,
                num_samples=num_samples,
            )
            results.append(exp_res)

            if out_dir:
                chart_path = out_dir / f"lime_sample_{i:04d}.png"
                self.plot_explanation(exp_res, save_path=chart_path)

        if out_dir:
            batch_json = out_dir / "lime_batch_results.json"
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved LIME batch results to {batch_json}")

        return results

    def explain_test_sample(
        self,
        n: int = 10,
        test_csv_path: Optional[Union[str, Path]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        num_features: int = 10,
        num_samples: int = 250,
        random_seed: int = 42,
        text_col: str = "cleaned_text",
        label_col: str = "normalized_label",
    ) -> List[Dict[str, Any]]:
        """
        Explain a stratified sample of n texts from the unified test set.

        Parameters
        ----------
        n : int
            Number of test samples to explain (default 10).
        test_csv_path : str or Path, optional
            Path to test.csv. Defaults to data/processed/unified/test.csv.
        save_dir : str or Path, optional
            Directory to save artifacts. Defaults to results/lime.
        num_features : int
            Top features to explain per text.
        num_samples : int
            LIME perturbation samples per explanation.
        random_seed : int
            Sampling and perturbation seed.
        text_col : str
            Column name with text.
        label_col : str
            Column name with integer label.

        Returns
        -------
        list of dict
            Enriched results with ground truth metadata.
        """
        import pandas as pd
        import csv

        if test_csv_path is None:
            test_csv_path = _get_project_root() / "data" / "processed" / "unified" / "test.csv"
        test_csv_path = Path(test_csv_path)

        if not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

        df = pd.read_csv(test_csv_path)
        logger.info(f"Loaded test set: {len(df)} rows from {test_csv_path}")

        # Resolve column names if standard variations exist
        if text_col not in df.columns and "text" in df.columns:
            text_col = "text"
        if label_col not in df.columns and "label" in df.columns:
            label_col = "label"

        # Stratified sampling
        rng = np.random.default_rng(random_seed)
        if label_col in df.columns:
            half = n // 2
            not_idx = df[df[label_col] == 0].index.tolist()
            hof_idx = df[df[label_col] == 1].index.tolist()
            sampled_not = rng.choice(not_idx, size=min(half, len(not_idx)), replace=False)
            sampled_hof = rng.choice(hof_idx, size=min(n - len(sampled_not), len(hof_idx)), replace=False)
            sample_idx = np.concatenate([sampled_not, sampled_hof])
            rng.shuffle(sample_idx)
            sample_df = df.loc[sample_idx].reset_index(drop=True)
        else:
            sample_df = df.sample(n=min(n, len(df)), random_state=random_seed).reset_index(drop=True)

        logger.info(f"Running LIME on {len(sample_df)} sampled test texts...")
        texts = sample_df[text_col].astype(str).tolist()

        if save_dir is None:
            save_dir = _get_project_root() / "results" / "lime"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, text in enumerate(texts):
            logger.info(f"Explaining sample {i+1}/{len(texts)}: {text[:40]}...")
            exp_res = self.explain_text(
                text=text,
                target_class=1,  # Explain Hate class
                num_features=num_features,
                num_samples=num_samples,
            )

            # Enrich with ground truth and metadata
            if label_col in sample_df.columns:
                gt_int = int(sample_df.loc[i, label_col])
                exp_res["ground_truth_label"] = self.label_names[gt_int]
                exp_res["ground_truth_int"] = gt_int
            if "language" in sample_df.columns:
                exp_res["language"] = str(sample_df.loc[i, "language"])
            if "dataset" in sample_df.columns:
                exp_res["dataset"] = str(sample_df.loc[i, "dataset"])

            results.append(exp_res)

            chart_path = save_dir / f"lime_sample_{i:04d}.png"
            self.plot_explanation(exp_res, save_path=chart_path)

        # Save JSON
        json_path = save_dir / "lime_test_sample_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved LIME test sample results to {json_path}")

        # Save Summary CSV
        summary_rows = []
        for r in results:
            pos_features = [f for f, w in r["raw_explanation"] if w > 0]
            neg_features = [f for f, w in r["raw_explanation"] if w < 0]
            summary_rows.append({
                "text": r["input_text"],
                "predicted_label": r["predicted_label"],
                "ground_truth_label": r.get("ground_truth_label", ""),
                "predicted_prob_hate": round(r["probability_hate"], 4),
                "language": r.get("language", ""),
                "dataset": r.get("dataset", ""),
                "top_hate_feature": pos_features[0] if pos_features else "",
                "top_non_hate_feature": neg_features[0] if neg_features else "",
                "surrogate_score": round(r["score"], 4),
            })

        csv_path = save_dir / "lime_test_sample_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Saved LIME test sample summary to {csv_path}")

        return results

    # ------------------------------------------------------------------
    # Visualization Helper
    # ------------------------------------------------------------------

    @staticmethod
    def plot_explanation(
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        top_k: int = 10,
    ):
        """
        Render a dark-themed horizontal bar chart of LIME feature weights.

        Red bars: positive weight (pushes toward explained class: Hate).
        Green bars: negative weight (pushes toward Non-Hate).

        Parameters
        ----------
        explanation : dict
            Result from explain_text().
        save_path : str or Path, optional
            Path to save figure. If None, figure is returned without saving.
        top_k : int
            Max features to display.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        features = explanation["features"][:top_k]
        weights = explanation["weights"][:top_k]
        target_name = explanation["explained_class_name"]
        prob_hate = explanation["probability_hate"]

        if not features:
            fig, ax = plt.subplots(figsize=(6, 2), dpi=120)
            ax.text(0.5, 0.5, "No LIME features found", ha="center", va="center", color="#cbd5e1")
            ax.set_facecolor("#0f172a")
            fig.patch.set_facecolor("#0f172a")
            if save_path:
                plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
                plt.close(fig)
            return fig

        # Reverse for top-down display in horizontal bar chart
        features_rev = features[::-1]
        weights_rev = weights[::-1]

        # Colors: red if pushes toward Hate, green if pushes toward Non-Hate
        if explanation["explained_class"] == 1:
            colors = ["#f43f5e" if w > 0 else "#10b981" for w in weights_rev]
        else:
            colors = ["#10b981" if w > 0 else "#f43f5e" for w in weights_rev]

        y_pos = np.arange(len(features_rev))
        fig_h = max(2.5, len(features_rev) * 0.35)

        fig, ax = plt.subplots(figsize=(6.5, fig_h), dpi=120)
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        ax.barh(y_pos, weights_rev, color=colors, height=0.6, edgecolor="none")
        ax.axvline(0, color="#475569", linewidth=1.0, linestyle="--")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_rev, fontsize=8.5, color="#cbd5e1")
        ax.set_xlabel(
            f"LIME Weight (impact on {target_name})",
            fontsize=8,
            color="#94a3b8",
        )
        ax.tick_params(colors="#94a3b8", labelsize=8)

        for spine in ax.spines.values():
            spine.set_color("#1e293b")

        ax.grid(axis="x", color="#1e293b", linestyle=":", linewidth=0.7)
        ax.set_title(
            f"LIME Feature Weights  —  Explained: {target_name.upper()}  "
            f"[P(Hate)={prob_hate:.3f}]",
            fontsize=9.5,
            color="#cbd5e1",
            pad=8,
        )

        fig.tight_layout(pad=0.9)

        if save_path:
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
            return None

        return fig
