"""
Robustness Evaluation Module (Review 2 Architecture)
======================================================
Provides tools for measuring cross-dataset generalization
and representation robustness.

CURRENT STATUS: Architecture prepared for Review 2.
No robustness experiments are run in Review 1.

The key metric is Robustness Drop:
    Robustness Drop = In-domain F1 - Out-of-domain F1

This quantifies how much performance degrades when the model
is applied outside its training distribution.

Usage (Review 2)
-----------------
    from src.evaluation.robustness import compute_robustness_drop
    drop = compute_robustness_drop(indomain_f1=0.88, outdomain_f1=0.72)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robustness Drop Metric
# ---------------------------------------------------------------------------

def compute_robustness_drop(
    indomain_f1: float,
    outdomain_f1: float,
) -> Dict:
    """
    Compute the robustness drop between in-domain and out-of-domain F1.

    Parameters
    ----------
    indomain_f1 : float
        Macro F1 on in-domain test set.
    outdomain_f1 : float
        Macro F1 on out-of-domain test set.

    Returns
    -------
    dict
        absolute_drop: float
        relative_drop_pct: float
        indomain_f1: float
        outdomain_f1: float
    """
    absolute_drop = indomain_f1 - outdomain_f1
    relative_drop = (absolute_drop / indomain_f1 * 100) if indomain_f1 > 0 else 0.0

    result = {
        "indomain_f1":      round(indomain_f1, 4),
        "outdomain_f1":     round(outdomain_f1, 4),
        "absolute_drop":    round(absolute_drop, 4),
        "relative_drop_pct": round(relative_drop, 2),
    }

    print(f"\n--- Robustness Analysis ---")
    print(f"  In-domain F1        : {indomain_f1:.4f}")
    print(f"  Out-of-domain F1    : {outdomain_f1:.4f}")
    print(f"  Absolute Drop       : {absolute_drop:.4f}")
    print(f"  Relative Drop       : {relative_drop:.2f}%")

    return result


# ---------------------------------------------------------------------------
# Representation Perturbation (Review 2)
# ---------------------------------------------------------------------------

def apply_spelling_variation(text: str, seed: int = 42) -> str:
    """
    Apply mild spelling variation to a text for robustness testing.

    IMPORTANT: This is NOT random corruption. It applies linguistically
    motivated, meaning-preserving variations commonly seen in
    code-mixed social media text.

    Variations applied:
    - Double a vowel (e.g., 'yaar' → 'yaaar')
    - Substitute 'ph' with 'f' or vice versa

    Parameters
    ----------
    text : str
    seed : int

    Returns
    -------
    str
        Perturbed text.

    Notes
    -----
    To be expanded in Review 2 after systematic study of common
    variation patterns in the actual dataset.
    """
    import re
    import random
    rng = random.Random(seed)

    variations = [
        (r'\byaar\b', 'yaaar'),
        (r'\bkya\b', 'kyaa'),
        (r'\bnahi\b', 'nahii'),
        (r'\bph', 'f'),
    ]

    chosen = rng.choice(variations)
    perturbed = re.sub(chosen[0], chosen[1], text)
    return perturbed


def generate_perturbation_set(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    n_samples: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a controlled perturbation test set from a sample of examples.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    text_col : str
        Column with text.
    n_samples : int
        Number of examples to perturb.
    seed : int

    Returns
    -------
    pd.DataFrame
        DataFrame with original and perturbed text columns.
        Clearly marked as perturbation test data.
    """
    sample = df.sample(n=min(n_samples, len(df)), random_state=seed).copy()
    sample["perturbed_text"] = sample[text_col].apply(
        lambda t: apply_spelling_variation(t, seed=seed)
    )
    sample["perturbation_type"] = "spelling_variation"
    sample["is_perturbation_data"] = True

    logger.info(
        f"Generated {len(sample)} perturbed examples for robustness testing."
    )
    return sample


# ---------------------------------------------------------------------------
# Cross-Dataset Experiment Summary (Review 2)
# ---------------------------------------------------------------------------

def summarize_cross_dataset_results(
    experiments: List[Dict],
) -> pd.DataFrame:
    """
    Build a cross-dataset robustness summary table.

    Parameters
    ----------
    experiments : list of dict
        Each dict should have keys:
            - model_name
            - train_dataset
            - test_dataset
            - macro_f1
            - is_indomain (bool)

    Returns
    -------
    pd.DataFrame
        Summary with robustness drops computed.
    """
    rows = []
    for exp in experiments:
        rows.append({
            "Model":         exp.get("model_name", ""),
            "Train Dataset": exp.get("train_dataset", ""),
            "Test Dataset":  exp.get("test_dataset", ""),
            "Macro F1":      exp.get("macro_f1", None),
            "In-Domain":     exp.get("is_indomain", False),
        })
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # For each model, compute robustness drop vs in-domain baseline
    result_rows = []
    for model in df["Model"].unique():
        m_df = df[df["Model"] == model]
        indomain_rows = m_df[m_df["In-Domain"] == True]
        if indomain_rows.empty:
            continue
        indomain_f1 = indomain_rows.iloc[0]["Macro F1"]
        for _, row in m_df.iterrows():
            drop = compute_robustness_drop(indomain_f1, row["Macro F1"])
            result_rows.append({
                **row.to_dict(),
                "Absolute Drop": drop["absolute_drop"],
                "Relative Drop %": drop["relative_drop_pct"],
            })

    return pd.DataFrame(result_rows)
