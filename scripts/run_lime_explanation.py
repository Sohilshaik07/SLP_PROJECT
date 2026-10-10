"""
Phase 3D: LIME Explanation Runner
==================================
Generates LIME token/word-level explanations for the fine-tuned XLM-R hate speech
classifier on a stratified sample of 10 texts from the unified test set.

Usage:
    python scripts/run_lime_explanation.py

Outputs (results/lime/):
    lime_test_sample_results.json   -- full LIME results for all samples
    lime_test_sample_summary.csv    -- per-sample summary table
    lime_sample_NNNN.png            -- individual bar charts
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Phase 3D — LIME Explanation Runner")
    logger.info("=" * 60)

    from src.explain.lime_explainer import XLMRLimeExplainer

    save_dir = PROJECT_ROOT / "results" / "lime"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model and build LIME explainer
    # ------------------------------------------------------------------
    logger.info("\n[1/3] Loading XLM-R model and LIME explainer...")
    explainer = XLMRLimeExplainer()
    explainer.load_model()

    # ------------------------------------------------------------------
    # 2. Explain 10 stratified samples from the test set
    # ------------------------------------------------------------------
    logger.info("\n[2/3] Running LIME on 10 stratified test samples...")
    results = explainer.explain_test_sample(
        n=10,
        save_dir=str(save_dir),
        num_samples=250,
        random_seed=42,
    )

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    logger.info("\n[3/3] Summary")
    logger.info("-" * 60)
    correct = sum(
        1 for r in results
        if r.get("predicted_label") == r.get("ground_truth_int")
    )
    logger.info(f"  Samples explained:   {len(results)}")
    logger.info(f"  Correct predictions: {correct}/{len(results)}")
    for r in results:
        gt = r.get("ground_truth_label", "?")
        pred = r["predicted_class_name"]
        prob = r["probability_hate"]
        status = "OK" if gt == pred else "MISCLASSIFIED"
        print(f"  [{status}] {r['input_text'][:45]:<45} | Pred: {pred} (P(Hate)={prob:.3f}) | True: {gt}")

    logger.info("-" * 60)
    logger.info(f"All LIME artifacts saved to: {save_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
