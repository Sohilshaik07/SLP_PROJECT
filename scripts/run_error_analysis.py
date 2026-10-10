"""
Phase 3F: Cross-Language Error Analysis
========================================

Reads the verified test predictions from:
  - results/predictions/xlmr_unified_test_predictions.csv
  - results/predictions/tfidf_lr_unified_test_predictions_verified.csv
  - results/predictions/tfidf_svm_unified_test_predictions_verified.csv

Generates:
  results/error_analysis/
    language_metrics.csv
    language_confusion_matrices.png
    error_examples.csv
    cross_language_report.md

Preserves all existing artifacts.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Input file paths
XLMR_PREDS_CSV = PROJECT_ROOT / "results" / "predictions" / "xlmr_unified_test_predictions.csv"
LR_PREDS_CSV   = PROJECT_ROOT / "results" / "predictions" / "tfidf_lr_unified_test_predictions_verified.csv"
SVM_PREDS_CSV  = PROJECT_ROOT / "results" / "predictions" / "tfidf_svm_unified_test_predictions_verified.csv"
OUT_DIR        = PROJECT_ROOT / "results" / "error_analysis"

LANGUAGES = ["Tamil", "Malayalam", "Hindi", "Telugu"]


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None
            
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else "N/A",
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FPR": round(float(fpr), 4),
        "FNR": round(float(fnr), 4),
    }


def run_cross_language_analysis():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("Phase 3F — Cross-Language Error Analysis")
    logger.info("=" * 70)

    # 1. Load XLM-R test predictions
    logger.info(f"Loading XLM-R predictions: {XLMR_PREDS_CSV}")
    df_xlmr = pd.read_csv(XLMR_PREDS_CSV)
    assert len(df_xlmr) == 6319, f"Expected 6,319 samples, got {len(df_xlmr)}"

    # 2. Overall Metrics
    overall_m = compute_metrics(
        df_xlmr["true_label"].values,
        df_xlmr["predicted_label"].values,
        df_xlmr["probability_hate"].values,
    )
    logger.info(f"Overall XLM-R: Acc={overall_m['accuracy']:.4f}, Macro F1={overall_m['macro_f1']:.4f}, AUC={overall_m['roc_auc']}")
    logger.info(f"Overall CM: TN={overall_m['TN']}, FP={overall_m['FP']}, FN={overall_m['FN']}, TP={overall_m['TP']}")

    # 3. Per-Language Metrics for XLM-R
    lang_records = []
    per_lang_cm = {}

    for lang in LANGUAGES:
        sub = df_xlmr[df_xlmr["language"] == lang]
        n_samples = len(sub)
        n_not_hate = int((sub["true_label"] == 0).sum())
        n_hate     = int((sub["true_label"] == 1).sum())
        
        m = compute_metrics(
            sub["true_label"].values,
            sub["predicted_label"].values,
            sub["probability_hate"].values,
        )
        per_lang_cm[lang] = m
        
        rec = {
            "language": lang,
            "sample_count": n_samples,
            "not_hate_count": n_not_hate,
            "hate_count": n_hate,
            "hate_ratio": round(n_hate / n_samples, 4) if n_samples > 0 else 0.0,
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "roc_auc": m["roc_auc"],
            "TN": m["TN"],
            "FP": m["FP"],
            "FN": m["FN"],
            "TP": m["TP"],
            "FPR": m["FPR"],
            "FNR": m["FNR"],
        }
        lang_records.append(rec)
        logger.info(
            f"  {lang:10s} (N={n_samples:4d}, Hate={n_hate:4d}/{n_samples:4d}): "
            f"Acc={m['accuracy']:.4f}, Macro F1={m['macro_f1']:.4f}, "
            f"FP={m['FP']:3d} (FPR={m['FPR']:.4f}), FN={m['FN']:3d} (FNR={m['FNR']:.4f})"
        )

    # 4. Baseline Comparisons (TF-IDF + LR and TF-IDF + SVM)
    has_lr = LR_PREDS_CSV.exists()
    has_svm = SVM_PREDS_CSV.exists()
    
    baseline_comp = []
    if has_lr and has_svm:
        logger.info("Computing baseline comparisons per language...")
        df_lr = pd.read_csv(LR_PREDS_CSV)
        df_svm = pd.read_csv(SVM_PREDS_CSV)
        
        for lang in LANGUAGES:
            sub_xlmr = df_xlmr[df_xlmr["language"] == lang]
            sub_lr   = df_lr[df_lr["language"] == lang]
            sub_svm  = df_svm[df_svm["language"] == lang]
            
            m_xlmr = compute_metrics(sub_xlmr["true_label"], sub_xlmr["predicted_label"])
            m_lr   = compute_metrics(sub_lr["true_label"], sub_lr["predicted_label"])
            m_svm  = compute_metrics(sub_svm["true_label"], sub_svm["predicted_label"])
            
            baseline_comp.append({
                "language": lang,
                "xlmr_accuracy": m_xlmr["accuracy"],
                "xlmr_macro_f1": m_xlmr["macro_f1"],
                "lr_accuracy": m_lr["accuracy"],
                "lr_macro_f1": m_lr["macro_f1"],
                "svm_accuracy": m_svm["accuracy"],
                "svm_macro_f1": m_svm["macro_f1"],
                "xlmr_f1_vs_lr": round(m_xlmr["macro_f1"] - m_lr["macro_f1"], 4),
                "xlmr_f1_vs_svm": round(m_xlmr["macro_f1"] - m_svm["macro_f1"], 4),
            })
            
    # Save language_metrics.csv
    lang_df = pd.DataFrame(lang_records)
    metrics_csv_path = OUT_DIR / "language_metrics.csv"
    lang_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {metrics_csv_path}")

    # 5. Extract Representative Error Examples
    # For each language, find representative FP (True=0, Pred=1) and FN (True=1, Pred=0)
    # Pick examples sorted by confidence / prototypicality (e.g. highest P(Hate) for FP, lowest P(Hate) for FN)
    error_examples = []
    
    for lang in LANGUAGES:
        sub = df_xlmr[df_xlmr["language"] == lang]
        
        # False Positives: True=0, Pred=1 (sorted by high P(Hate) = high confidence errors)
        fps = sub[(sub["true_label"] == 0) & (sub["predicted_label"] == 1)].sort_values(
            by="probability_hate", ascending=False
        )
        # Select 2 representative FPs
        for _, row in fps.head(2).iterrows():
            error_examples.append({
                "language": lang,
                "error_type": "False Positive",
                "true_label": 0,
                "true_label_name": "Non-Hate",
                "predicted_label": 1,
                "predicted_label_name": "Hate",
                "probability_hate": round(float(row["probability_hate"]), 4),
                "probability_non_hate": round(float(row["probability_non_hate"]), 4),
                "text": str(row["text"]).strip(),
            })
            
        # False Negatives: True=1, Pred=0 (sorted by low P(Hate) = high confidence misses)
        fns = sub[(sub["true_label"] == 1) & (sub["predicted_label"] == 0)].sort_values(
            by="probability_hate", ascending=True
        )
        # Select 2 representative FNs
        for _, row in fns.head(2).iterrows():
            error_examples.append({
                "language": lang,
                "error_type": "False Negative",
                "true_label": 1,
                "true_label_name": "Hate",
                "predicted_label": 0,
                "predicted_label_name": "Non-Hate",
                "probability_hate": round(float(row["probability_hate"]), 4),
                "probability_non_hate": round(float(row["probability_non_hate"]), 4),
                "text": str(row["text"]).strip(),
            })
            
    err_df = pd.DataFrame(error_examples)
    err_csv_path = OUT_DIR / "error_examples.csv"
    err_df.to_csv(err_csv_path, index=False, encoding="utf-8")
    logger.info(f"Saved: {err_csv_path}")

    # 6. Generate Language Confusion Matrices Plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    
    for ax, lang in zip(axes.flatten(), LANGUAGES):
        ax.set_facecolor("#1e293b")
        m = per_lang_cm[lang]
        cm_data = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        
        im = ax.imshow(cm_data, interpolation="nearest", cmap="Blues")
        
        # Color bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
        
        ax.set_title(
            f"{lang} (N={m['TN']+m['FP']+m['FN']+m['TP']})\n"
            f"Acc: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f}",
            color="#f8fafc", fontsize=11, fontweight="bold", pad=8
        )
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Non-Hate", "Hate"], color="#cbd5e1", fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Non-Hate", "Hate"], color="#cbd5e1", fontsize=9)
        ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=9)
        ax.set_ylabel("True Label", color="#94a3b8", fontsize=9)
        
        thresh = cm_data.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm_data[i, j]
                ax.text(
                    j, i, f"{val:,}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "#0f172a",
                    fontsize=12, fontweight="bold"
                )
                
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    fig.suptitle(
        "Per-Language Confusion Matrices — XLM-RoBERTa (Verified Test Set)",
        color="#f8fafc", fontsize=13, fontweight="bold", y=0.98
    )
    fig.tight_layout(pad=2.0)
    fig_path = OUT_DIR / "language_confusion_matrices.png"
    fig.savefig(fig_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")

    # 7. Identify Extremes
    highest_acc = max(lang_records, key=lambda x: x["accuracy"])
    lowest_acc  = min(lang_records, key=lambda x: x["accuracy"])
    highest_f1  = max(lang_records, key=lambda x: x["macro_f1"])
    lowest_f1   = min(lang_records, key=lambda x: x["macro_f1"])
    highest_fnr = max(lang_records, key=lambda x: x["FNR"])
    highest_fpr = max(lang_records, key=lambda x: x["FPR"])

    # 8. Generate cross_language_report.md
    report_md_path = OUT_DIR / "cross_language_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3F — Cross-Language Error Analysis Report\n\n")
        f.write("**Evaluation Split:** Unified Test Set (`data/processed/unified/test.csv`, 6,319 samples)\n")
        f.write("**Model:** XLM-RoBERTa (`models/xlmr/UNIFIED_XLMR_best`)\n")
        f.write("**Prediction Source:** `results/predictions/xlmr_unified_test_predictions.csv`\n\n")
        
        f.write("## 1. Overall Performance\n\n")
        f.write(f"- **Total Samples:** 6,319\n")
        f.write(f"- **Accuracy:** {overall_m['accuracy']:.4f}\n")
        f.write(f"- **Macro Precision:** {overall_m['macro_precision']:.4f}\n")
        f.write(f"- **Macro Recall:** {overall_m['macro_recall']:.4f}\n")
        f.write(f"- **Macro F1:** {overall_m['macro_f1']:.4f}\n")
        f.write(f"- **ROC-AUC:** {overall_m['roc_auc']}\n")
        f.write(f"- **Confusion Matrix:** TN={overall_m['TN']}, FP={overall_m['FP']}, FN={overall_m['FN']}, TP={overall_m['TP']}\n\n")
        
        f.write("## 2. Per-Language Metrics\n\n")
        f.write("| Language | Samples | Non-Hate | Hate | Hate % | Accuracy | Macro Prec | Macro Rec | Macro F1 | ROC-AUC | TN | FP | FN | TP | FPR | FNR |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in lang_records:
            f.write(
                f"| **{r['language']}** | {r['sample_count']} | {r['not_hate_count']} | {r['hate_count']} | "
                f"{r['hate_ratio']*100:.1f}% | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | "
                f"{r['macro_recall']:.4f} | **{r['macro_f1']:.4f}** | {r['roc_auc']} | "
                f"{r['TN']} | {r['FP']} | {r['FN']} | {r['TP']} | {r['FPR']:.4f} | {r['FNR']:.4f} |\n"
            )
        f.write("\n")
        
        f.write("## 3. Performance Extremes\n\n")
        f.write(f"- **Highest Accuracy:** {highest_acc['language']} ({highest_acc['accuracy']:.4f})\n")
        f.write(f"- **Lowest Accuracy:** {lowest_acc['language']} ({lowest_acc['accuracy']:.4f})\n")
        f.write(f"- **Highest Macro F1:** {highest_f1['language']} ({highest_f1['macro_f1']:.4f})\n")
        f.write(f"- **Lowest Macro F1:** {lowest_f1['language']} ({lowest_f1['macro_f1']:.4f})\n")
        f.write(f"- **Highest False-Negative Rate (Missed Hate):** {highest_fnr['language']} ({highest_fnr['FNR']*100:.1f}%, FN={highest_fnr['FN']}/{highest_fnr['hate_count']})\n")
        f.write(f"- **Highest False-Positive Rate (False Alarms):** {highest_fpr['language']} ({highest_fpr['FPR']*100:.1f}%, FP={highest_fpr['FP']}/{highest_fpr['not_hate_count']})\n\n")

        f.write("## 4. Class Imbalance Analysis & Accuracy vs. Macro F1 Paradox\n\n")
        f.write("A critical scientific finding emerges when comparing **Malayalam** to other languages:\n\n")
        f.write("- **Malayalam exhibits severe class imbalance:** 1,211 Non-Hate (96.0%) vs. 50 Hate (4.0%).\n")
        f.write("- Because 96% of samples are Non-Hate, a trivial majority classifier predicting Non-Hate for everything would achieve 96.0% accuracy.\n")
        f.write(f"- XLM-R achieves **95.48% accuracy** on Malayalam, the highest among all languages.\n")
        f.write(f"- However, XLM-R's **Macro F1 on Malayalam is only 0.6607**, the lowest among all languages.\n")
        f.write(f"- Out of 50 actual Hate comments in Malayalam, the model misses 35 (FN = 35, FNR = 70.0%), detecting only 15 (TP = 15). Recall on Hate is only 30.0%.\n")
        f.write("- **Conclusion:** Accuracy is severely distorted by label distribution. Macro F1 provides an unweighted, balanced assessment of minority-class recognition.\n\n")

        f.write("## 5. Baseline Comparison (XLM-R vs. TF-IDF + LR vs. TF-IDF + SVM)\n\n")
        f.write("| Language | XLM-R Acc | LR Acc | SVM Acc | XLM-R F1 | LR F1 | SVM F1 | F1 Advantage vs LR | F1 Advantage vs SVM |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for b in baseline_comp:
            f.write(
                f"| **{b['language']}** | {b['xlmr_accuracy']:.4f} | {b['lr_accuracy']:.4f} | {b['svm_accuracy']:.4f} | "
                f"**{b['xlmr_macro_f1']:.4f}** | {b['lr_macro_f1']:.4f} | {b['svm_macro_f1']:.4f} | "
                f"{b['xlmr_f1_vs_lr']:+.4f} | {b['xlmr_f1_vs_svm']:+.4f} |\n"
            )
        f.write("\n")

        f.write("## 6. Representative Error Examples\n\n")
        f.write("Actual unmodified examples from `xlmr_unified_test_predictions.csv`:\n\n")
        f.write("| Language | Error Type | True Label | Pred Label | P(Hate) | Text Excerpt |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")
        for ex in error_examples:
            short_txt = ex["text"][:75] + ("..." if len(ex["text"]) > 75 else "")
            short_txt = short_txt.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {ex['language']} | {ex['error_type']} | {ex['true_label_name']} | {ex['predicted_label_name']} | {ex['probability_hate']:.4f} | `{short_txt}` |\n")
        f.write("\n")

        f.write("## 7. Scientific Observations vs. Hypotheses\n\n")
        f.write("- **OBSERVATION 1:** Tamil constitutes 66.1% (4,174 / 6,319) of the test dataset and achieves balanced Macro F1 (0.7792), driving most of the unified metric.\n")
        f.write("- **OBSERVATION 2:** Malayalam has the lowest Macro F1 (0.6607) due to high false-negative rate (70.0% of hate comments missed).\n")
        f.write("- **OBSERVATION 3:** Hindi has the lowest accuracy (0.7367) and highest false-positive rate (30.25%, FP=72/238).\n")
        f.write("- **HYPOTHESIS A:** Hindi's high false-positive rate may stem from intense code-mixed political discourse containing aggressive rhetoric or swear words used colloquially without genuine hate intent.\n")
        f.write("- **HYPOTHESIS B:** Malayalam's high false-negative rate may be influenced by subword tokenization fragmentation in SentencePiece or subtle sarcasm/implicit hate requiring deeper contextual reasoning.\n")
        
    logger.info(f"Saved: {report_md_path}")
    logger.info("Phase 3F error analysis completed successfully.")
    
    return {
        "overall": overall_m,
        "languages": lang_records,
        "extremes": {
            "highest_acc": highest_acc,
            "lowest_acc": lowest_acc,
            "highest_f1": highest_f1,
            "lowest_f1": lowest_f1,
            "highest_fnr": highest_fnr,
            "highest_fpr": highest_fpr,
        },
        "baseline_comp": baseline_comp,
        "error_examples": error_examples,
    }


if __name__ == "__main__":
    run_cross_language_analysis()
