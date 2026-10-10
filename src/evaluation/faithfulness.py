"""
Phase 3E: Faithfulness Evaluation of SHAP and LIME
====================================================

Evaluates whether features identified as important by SHAP and LIME correspond
to meaningful changes in the XLM-R model's predicted probability, compared to
a random-deletion baseline.

Method: Deletion-based feature attribution faithfulness.

For each sample:
  - Obtain SHAP top-k features and LIME top-k features
  - Delete those features from the input text
  - Re-run the SAME XLM-R model on the modified text
  - Record probability change vs. original

Design principles:
  - READ-ONLY: Model weights are never modified.
  - NO HARD-CODING: All results are generated from live model inference.
  - HONEST REPORTING: Results are reported as-is, whether SHAP/LIME outperform
    random or not. No tuning to produce preferred outcomes.
  - INDEPENDENCE: SHAP and LIME explanations are generated independently.
  - DOCUMENTED LIMITATIONS: Local surrogate faithfulness; not causal proof.

Outputs:
  results/faithfulness/
    faithfulness_results.json        -- per-sample, per-method, per-k results
    faithfulness_summary.csv         -- aggregate metrics by method and k
    faithfulness_by_sample.csv       -- per-sample detail
    faithfulness_probability_drop.png    -- bar chart: mean drop by method/k
    faithfulness_method_comparison.png   -- comparison chart
"""

import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLE_SEED      = 99         # Different from Phase 3B seed=42
SHAP_EVALS       = 250        # SHAP max_evals per explanation
LIME_SAMPLES     = 200        # LIME num_samples per explanation
RANDOM_SEED_BASE = 77         # Seed for random-deletion baseline
K_VALUES         = [1, 3, 5]  # Feature deletion depths
N_SAMPLES        = 20         # Total samples (10 NOT-HATE + 10 HATE)
BOOTSTRAP_N      = 1000       # Bootstrap resampling iterations for CIs
BOOTSTRAP_SEED   = 42         # Deterministic bootstrap seed

MODEL_PATH  = (PROJECT_ROOT / "models" / "xlmr" / "UNIFIED_XLMR_best").resolve()
TEST_CSV    = PROJECT_ROOT / "data" / "processed" / "unified" / "test.csv"
SHAP_CSV    = PROJECT_ROOT / "results" / "shap" / "shap_test_sample_summary.csv"
OUT_DIR     = PROJECT_ROOT / "results" / "faithfulness"
LABEL_NAMES = ["Non-Hate", "Hate"]


# ============================================================================
# DELETION HELPERS
# ============================================================================

def _delete_shap_features(text: str, shap_tokens: list, shap_values: list, k: int) -> str:
    """
    Remove the top-k SHAP tokens from the text by absolute SHAP value.

    For multilingual / Indic code-mixed text, tokens are removed by exact
    substring replacement of each selected token (or stripped token), avoiding
    ASCII \\b regex word-boundary failures on non-Latin scripts.

    Parameters
    ----------
    text : str
        Original input text.
    shap_tokens : list of str
        Tokens returned by XLMRShapExplainer.
    shap_values : list of float
        SHAP values for the HOF class (same index as shap_tokens).
    k : int
        Number of top-magnitude tokens to remove.

    Returns
    -------
    str
        Modified text with top-k tokens removed.
    """
    if not shap_tokens or k <= 0:
        return text

    abs_vals = [abs(v) for v in shap_values]
    top_idx = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i], reverse=True)[:k]
    to_remove = [shap_tokens[i] for i in top_idx]

    modified = text
    for tok in to_remove:
        tok_clean = tok.strip()
        if tok_clean and tok_clean in modified:
            modified = modified.replace(tok_clean, "", 1)
        elif tok in modified:
            modified = modified.replace(tok, "", 1)

    modified = re.sub(r'\s+', ' ', modified).strip()
    return modified if modified else " "


def _delete_lime_features(text: str, lime_features: list, lime_weights: list, k: int) -> str:
    """
    Remove the top-k LIME word features from the text by absolute LIME weight.

    Replaces features by exact string replacement, avoiding regex word-boundary
    issues on Indic scripts.

    Parameters
    ----------
    text : str
        Original input text.
    lime_features : list of str
        Word features from XLMRLimeExplainer.
    lime_weights : list of float
        LIME weights for the Hate class.
    k : int
        Number of top-magnitude features to remove.

    Returns
    -------
    str
        Modified text with top-k words removed.
    """
    if not lime_features or k <= 0:
        return text

    abs_vals = [abs(w) for w in lime_weights]
    top_idx = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i], reverse=True)[:k]
    to_remove = [str(lime_features[i]).strip() for i in top_idx if str(lime_features[i]).strip()]

    modified = text
    for feat in to_remove:
        if feat in modified:
            modified = modified.replace(feat, "", 1)

    modified = re.sub(r'\s+', ' ', modified).strip()
    return modified if modified else " "


def _delete_random_features(text: str, candidate_features: list, k: int, seed: int) -> str:
    """
    Remove k randomly selected features from candidate_features found in text.

    Uses the same exact-match deletion logic as LIME for a fair comparison.
    A deterministic seed is used for reproducibility.

    Parameters
    ----------
    text : str
        Original input text.
    candidate_features : list of str
        Pool of eligible feature tokens (words from the text).
    k : int
        Number to randomly select.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Modified text with k random words removed.
    """
    rng = np.random.default_rng(seed)
    valid = [f for f in candidate_features if f.strip() and f.strip() in text]
    if not valid:
        return text
    k_eff = min(k, len(valid))
    chosen = list(rng.choice(valid, size=k_eff, replace=False))

    modified = text
    for feat in chosen:
        feat_s = feat.strip()
        if feat_s in modified:
            modified = modified.replace(feat_s, "", 1)

    modified = re.sub(r'\s+', ' ', modified).strip()
    return modified if modified else " "


# ============================================================================
# BOOTSTRAP CI
# ============================================================================

def _bootstrap_ci(values: list, n: int = 1000, seed: int = 42, ci: float = 0.95) -> tuple:
    """Return (mean, lower_ci, upper_ci) via percentile bootstrap."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng  = np.random.default_rng(seed)
    arr  = np.array(values, dtype=float)
    mns  = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.mean(arr)),
        float(np.percentile(mns, alpha * 100)),
        float(np.percentile(mns, (1 - alpha) * 100)),
    )


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_faithfulness_evaluation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Phase 3E — Faithfulness Evaluation of SHAP and LIME")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    logger.info("\n[1/6] Loading XLM-R model (read-only, weights NOT modified)...")
    from src.utils.config import load_config
    from src.models.xlmr_model import XLMRClassifier
    from src.explain.shap_explainer import XLMRShapExplainer
    from src.explain.lime_explainer import XLMRLimeExplainer

    cfg = load_config()
    classifier = XLMRClassifier(config=cfg)
    classifier.load(str(MODEL_PATH))
    logger.info("XLMRClassifier loaded successfully.")

    shap_exp = XLMRShapExplainer()
    shap_exp.load_from_classifier(classifier)
    logger.info("XLMRShapExplainer built (no second disk load).")

    lime_exp = XLMRLimeExplainer(random_state=42)
    lime_exp.load_from_classifier(classifier)
    logger.info("XLMRLimeExplainer built (no second disk load).")

    # ------------------------------------------------------------------
    # 2. Select Phase 3E samples (no overlap with Phase 3B)
    # ------------------------------------------------------------------
    logger.info("\n[2/6] Selecting 20 stratified test samples (seed=99, no Phase 3B overlap)...")
    test_df   = pd.read_csv(TEST_CSV)
    shap_csv  = pd.read_csv(SHAP_CSV)
    used_texts = set(shap_csv["text"].astype(str).tolist())
    phase3b_idx = set(test_df[test_df["text"].isin(used_texts)].index.tolist())

    rng = np.random.default_rng(SAMPLE_SEED)
    not_idx = [i for i in test_df[test_df["label"] == 0].index if i not in phase3b_idx]
    hof_idx = [i for i in test_df[test_df["label"] == 1].index if i not in phase3b_idx]
    s_not = rng.choice(not_idx, size=N_SAMPLES // 2, replace=False)
    s_hof = rng.choice(hof_idx, size=N_SAMPLES // 2, replace=False)
    sample_idx = np.concatenate([s_not, s_hof])
    rng.shuffle(sample_idx)
    sample_df = test_df.loc[sample_idx].reset_index(drop=True)

    logger.info(f"  Samples: {len(sample_df)}  |  NOT-HATE: {(sample_df['label']==0).sum()}  |  HATE: {(sample_df['label']==1).sum()}")
    logger.info(f"  Languages: {dict(sample_df['language'].value_counts())}")
    logger.info(f"  Phase 3B overlap: {len(set(sample_idx) & phase3b_idx)}")

    # ------------------------------------------------------------------
    # 3. Preliminary deletion verification (3 samples)
    # ------------------------------------------------------------------
    logger.info("\n[3/6] Preliminary deletion verification (3 samples)...")
    for vi in range(min(3, len(sample_df))):
        vtext = str(sample_df.loc[vi, "text"])
        _, vprobs = classifier.predict([vtext])
        vorig_prob = float(vprobs[0][1])  # P(Hate)

        # Simple SHAP-based deletion preview (1 feature)
        vshap = shap_exp.explain_text(vtext, max_evals=150)
        vmod  = _delete_shap_features(vtext, vshap["tokens"], vshap["shap_values_hof"], k=1)
        _, vmod_probs = classifier.predict([vmod])
        vmod_prob = float(vmod_probs[0][1])

        assert vmod != vtext or len(vshap["tokens"]) == 0, \
            f"Deleted text must differ from original (sample {vi})"
        assert isinstance(vorig_prob, float), "Original probability must be float"
        assert isinstance(vmod_prob, float), "Modified probability must be float"
        logger.info(f"  Sample {vi}: orig_P(Hate)={vorig_prob:.4f} → modified_P(Hate)={vmod_prob:.4f} | deleted text differs: {vmod != vtext}")

    # ------------------------------------------------------------------
    # 4. Full faithfulness experiment
    # ------------------------------------------------------------------
    logger.info(f"\n[4/6] Running faithfulness experiment on {N_SAMPLES} samples, k={K_VALUES}...")
    all_results = []

    for si, row in sample_df.iterrows():
        text  = str(row["text"])
        label = int(row["label"])
        lang  = str(row.get("language", ""))
        ds    = str(row.get("dataset", ""))

        logger.info(f"\n  Sample {si+1}/{N_SAMPLES}: [{lang}] label={label} | {text[:45]}...")

        # Original prediction
        _, orig_probs = classifier.predict([text])
        orig_pred = int(np.argmax(orig_probs[0]))
        orig_prob_class = float(orig_probs[0][orig_pred])  # P(original predicted class)
        orig_prob_hate  = float(orig_probs[0][1])

        # Generate SHAP explanation
        logger.info("    Generating SHAP explanation...")
        shap_res = shap_exp.explain_text(text, max_evals=SHAP_EVALS)
        shap_tokens = shap_res["tokens"]
        shap_vals   = shap_res["shap_values_hof"]

        # Generate LIME explanation
        logger.info("    Generating LIME explanation...")
        lime_res  = lime_exp.explain_text(text, target_class=1, num_features=15, num_samples=LIME_SAMPLES)
        lime_feats = lime_res["features"]
        lime_wts   = lime_res["weights"]

        # Random candidate feature pool = words in text (LIME's feature space)
        word_pool = [w.strip() for w in re.split(r'[\s\.,!?;:\"\'\(\)\[\]]+', text) if w.strip()]
        word_pool = list(dict.fromkeys(word_pool)) if word_pool else [text]

        sample_record = {
            "sample_id": int(si),
            "text": text,
            "language": lang,
            "dataset": ds,
            "true_label": label,
            "true_label_name": LABEL_NAMES[label],
            "predicted_label": orig_pred,
            "predicted_class_name": LABEL_NAMES[orig_pred],
            "orig_prob_predicted_class": round(orig_prob_class, 6),
            "orig_prob_hate": round(orig_prob_hate, 6),
            "shap_tokens": shap_tokens,
            "shap_values_hof": [round(v, 6) for v in shap_vals],
            "lime_features": lime_feats,
            "lime_weights": [round(w, 6) for w in lime_wts],
            "deletion_results": [],
        }

        for k in K_VALUES:
            # -- SHAP deletion --
            shap_del_text = _delete_shap_features(text, shap_tokens, shap_vals, k)
            _, shap_del_probs = classifier.predict([shap_del_text])
            shap_del_prob = float(shap_del_probs[0][orig_pred])
            shap_drop = orig_prob_class - shap_del_prob

            # -- LIME deletion --
            lime_del_text = _delete_lime_features(text, lime_feats, lime_wts, k)
            _, lime_del_probs = classifier.predict([lime_del_text])
            lime_del_prob = float(lime_del_probs[0][orig_pred])
            lime_drop = orig_prob_class - lime_del_prob

            # -- Random deletion baseline (deterministic per-sample seed) --
            rand_seed = RANDOM_SEED_BASE + int(si) * 100 + k
            rand_del_text = _delete_random_features(text, word_pool, k, rand_seed)
            _, rand_del_probs = classifier.predict([rand_del_text])
            rand_del_prob = float(rand_del_probs[0][orig_pred])
            rand_drop = orig_prob_class - rand_del_prob

            logger.info(f"    k={k}: SHAP_drop={shap_drop:+.4f} | LIME_drop={lime_drop:+.4f} | Rand_drop={rand_drop:+.4f}")

            # Determine which SHAP features were actually selected
            abs_shap = [abs(v) for v in shap_vals]
            top_shap_idx = sorted(range(len(abs_shap)), key=lambda i: abs_shap[i], reverse=True)[:k]
            selected_shap = [shap_tokens[i].strip() for i in top_shap_idx if shap_tokens[i].strip()]

            abs_lime = [abs(w) for w in lime_wts]
            top_lime_idx = sorted(range(len(abs_lime)), key=lambda i: abs_lime[i], reverse=True)[:k]
            selected_lime = [lime_feats[i] for i in top_lime_idx]

            sample_record["deletion_results"].append({
                "k": k,
                "shap": {
                    "selected_features": selected_shap,
                    "deleted_text": shap_del_text,
                    "orig_prob": round(orig_prob_class, 6),
                    "modified_prob": round(shap_del_prob, 6),
                    "probability_drop": round(shap_drop, 6),
                    "decreased": bool(shap_drop > 0),
                },
                "lime": {
                    "selected_features": selected_lime,
                    "deleted_text": lime_del_text,
                    "orig_prob": round(orig_prob_class, 6),
                    "modified_prob": round(lime_del_prob, 6),
                    "probability_drop": round(lime_drop, 6),
                    "decreased": bool(lime_drop > 0),
                },
                "random": {
                    "seed": rand_seed,
                    "deleted_text": rand_del_text,
                    "orig_prob": round(orig_prob_class, 6),
                    "modified_prob": round(rand_del_prob, 6),
                    "probability_drop": round(rand_drop, 6),
                    "decreased": bool(rand_drop > 0),
                },
            })

        all_results.append(sample_record)

    # ------------------------------------------------------------------
    # 5. Aggregate metrics
    # ------------------------------------------------------------------
    logger.info("\n[5/6] Computing aggregate metrics...")

    # Collect drops per method per k
    drops = {
        "shap":   {k: [] for k in K_VALUES},
        "lime":   {k: [] for k in K_VALUES},
        "random": {k: [] for k in K_VALUES},
    }
    decreased_fracs = {m: {k: [] for k in K_VALUES} for m in ["shap", "lime", "random"]}

    for rec in all_results:
        for dr in rec["deletion_results"]:
            k = dr["k"]
            drops["shap"][k].append(dr["shap"]["probability_drop"])
            drops["lime"][k].append(dr["lime"]["probability_drop"])
            drops["random"][k].append(dr["random"]["probability_drop"])
            decreased_fracs["shap"][k].append(int(dr["shap"]["decreased"]))
            decreased_fracs["lime"][k].append(int(dr["lime"]["decreased"]))
            decreased_fracs["random"][k].append(int(dr["random"]["decreased"]))

    summary_rows = []
    for method in ["shap", "lime", "random"]:
        for k in K_VALUES:
            vals = drops[method][k]
            mean_v, ci_lo, ci_hi = _bootstrap_ci(vals, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            row = {
                "method": method.upper(),
                "k": k,
                "n_samples": len(vals),
                "mean_prob_drop": round(mean_v, 6),
                "median_prob_drop": round(float(np.median(vals)), 6),
                "mean_abs_change": round(float(np.mean([abs(v) for v in vals])), 6),
                "frac_decreased": round(float(np.mean(decreased_fracs[method][k])), 4),
                "bootstrap_95ci_lo": round(ci_lo, 6),
                "bootstrap_95ci_hi": round(ci_hi, 6),
            }
            summary_rows.append(row)
            logger.info(
                f"  {method.upper():6s} k={k}: mean_drop={mean_v:+.4f} "
                f"[95%CI {ci_lo:+.4f}, {ci_hi:+.4f}] "
                f"frac_dec={row['frac_decreased']:.2f}"
            )

    # SHAP/LIME advantage over random
    advantage_rows = []
    for method in ["shap", "lime"]:
        for k in K_VALUES:
            adv = float(np.mean(drops[method][k])) - float(np.mean(drops["random"][k]))
            logger.info(f"  {method.upper()} vs Random k={k}: advantage = {adv:+.4f}")
            advantage_rows.append({"method": method.upper(), "k": k, "advantage_over_random": round(adv, 6)})

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Saving faithfulness artifacts...")

    # a) Full JSON
    meta = {
        "experiment": "Phase 3E — Faithfulness Evaluation",
        "model_path": str(MODEL_PATH),
        "test_csv": str(TEST_CSV),
        "sample_seed": SAMPLE_SEED,
        "sample_count": N_SAMPLES,
        "shap_max_evals": SHAP_EVALS,
        "lime_num_samples": LIME_SAMPLES,
        "random_seed_base": RANDOM_SEED_BASE,
        "k_values": K_VALUES,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "phase3b_overlap": 0,
        "summary": summary_rows,
        "advantage": advantage_rows,
        "samples": all_results,
    }
    json_path = OUT_DIR / "faithfulness_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved: {json_path}")

    # b) Summary CSV
    summary_csv_path = OUT_DIR / "faithfulness_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"  Saved: {summary_csv_path}")

    # c) Per-sample CSV
    per_sample_rows = []
    for rec in all_results:
        for dr in rec["deletion_results"]:
            k = dr["k"]
            per_sample_rows.append({
                "sample_id": rec["sample_id"],
                "language": rec["language"],
                "true_label_name": rec["true_label_name"],
                "predicted_class_name": rec["predicted_class_name"],
                "k": k,
                "orig_prob": dr["shap"]["orig_prob"],
                "shap_modified_prob": dr["shap"]["modified_prob"],
                "shap_drop": dr["shap"]["probability_drop"],
                "shap_decreased": int(dr["shap"]["decreased"]),
                "lime_modified_prob": dr["lime"]["modified_prob"],
                "lime_drop": dr["lime"]["probability_drop"],
                "lime_decreased": int(dr["lime"]["decreased"]),
                "rand_modified_prob": dr["random"]["modified_prob"],
                "rand_drop": dr["random"]["probability_drop"],
                "rand_decreased": int(dr["random"]["decreased"]),
                "shap_selected": "|".join(dr["shap"]["selected_features"]),
                "lime_selected": "|".join(dr["lime"]["selected_features"]),
            })
    per_sample_csv_path = OUT_DIR / "faithfulness_by_sample.csv"
    with open(per_sample_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    logger.info(f"  Saved: {per_sample_csv_path}")

    # d) Charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Mean probability drop by method and k
    methods_plot = ["SHAP", "LIME", "RANDOM"]
    colors_plot  = ["#818cf8", "#f43f5e", "#94a3b8"]
    x = np.arange(len(K_VALUES))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for mi, (method, color) in enumerate(zip(methods_plot, colors_plot)):
        means  = [r["mean_prob_drop"] for r in summary_rows if r["method"] == method]
        ci_los = [r["bootstrap_95ci_lo"] for r in summary_rows if r["method"] == method]
        ci_his = [r["bootstrap_95ci_hi"] for r in summary_rows if r["method"] == method]
        errs   = [[m - lo for m, lo in zip(means, ci_los)],
                  [hi - m for m, hi in zip(means, ci_his)]]
        ax.bar(x + mi * bar_w, means, width=bar_w, label=method, color=color,
               edgecolor="none", alpha=0.88)
        ax.errorbar(x + mi * bar_w, means, yerr=errs, fmt="none",
                    ecolor="#475569", elinewidth=1.0, capsize=3)

    ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([f"k={k}" for k in K_VALUES], color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Mean Probability Drop (original class)", color="#94a3b8", fontsize=9)
    ax.set_title("Faithfulness: Mean Probability Drop by Method and k\n"
                 "(Higher = more faithful; 95% bootstrap CI shown)",
                 color="#cbd5e1", fontsize=10, pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.legend(facecolor="#0f172a", labelcolor="#cbd5e1", edgecolor="#334155", fontsize=9)
    ax.grid(axis="y", color="#1e293b", linestyle=":", linewidth=0.7)
    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_DIR / "faithfulness_probability_drop.png",
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {OUT_DIR / 'faithfulness_probability_drop.png'}")

    # Plot 2: Fraction of samples with decreased probability
    fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=120)
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")
    for mi, (method, color) in enumerate(zip(methods_plot, colors_plot)):
        fracs = [r["frac_decreased"] for r in summary_rows if r["method"] == method]
        ax2.plot([f"k={k}" for k in K_VALUES], fracs, marker="o",
                 label=method, color=color, linewidth=2.0, markersize=6)
    ax2.axhline(0.5, color="#475569", linewidth=0.8, linestyle="--", label="50% baseline")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fraction of Samples with Probability Decreased", color="#94a3b8", fontsize=9)
    ax2.set_title("Faithfulness: Fraction of Samples where P(original class) Decreased after Deletion",
                  color="#cbd5e1", fontsize=10, pad=10)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#1e293b")
    ax2.legend(facecolor="#0f172a", labelcolor="#cbd5e1", edgecolor="#334155", fontsize=9)
    ax2.grid(color="#1e293b", linestyle=":", linewidth=0.7)
    fig2.tight_layout(pad=1.0)
    fig2.savefig(OUT_DIR / "faithfulness_method_comparison.png",
                 facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved: {OUT_DIR / 'faithfulness_method_comparison.png'}")

    # ------------------------------------------------------------------
    # 7. Print final report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3E FAITHFULNESS REPORT")
    logger.info("=" * 70)
    logger.info(f"Sample count: {N_SAMPLES}  (NOT-HATE: 10, HATE: 10)")
    logger.info(f"Phase 3B overlap: 0 (seed={SAMPLE_SEED} vs. Phase 3B seed=42)")
    logger.info(f"k values: {K_VALUES}")
    logger.info(f"Bootstrap CI: 95%, n={BOOTSTRAP_N} resamples, seed={BOOTSTRAP_SEED}")
    logger.info("")
    for row in summary_rows:
        logger.info(
            f"  {row['method']:6s} k={row['k']}: "
            f"mean_drop={row['mean_prob_drop']:+.4f} "
            f"[{row['bootstrap_95ci_lo']:+.4f}, {row['bootstrap_95ci_hi']:+.4f}] "
            f"median={row['median_prob_drop']:+.4f} "
            f"frac_dec={row['frac_decreased']:.2f}"
        )
    logger.info("")
    for row in advantage_rows:
        logger.info(f"  {row['method']} vs Random k={row['k']}: advantage = {row['advantage_over_random']:+.4f}")
    logger.info("=" * 70)

    return summary_rows, advantage_rows, all_results


if __name__ == "__main__":
    run_faithfulness_evaluation()
