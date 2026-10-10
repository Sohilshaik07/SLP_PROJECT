"""
Phase 3B: SHAP Explanation Runner
===================================
Generates SHAP token-level explanations for the XLM-R hate speech classifier
on a stratified sample of 20 texts from the unified test set.

Usage:
    python scripts/run_shap_explanation.py

Outputs (results/shap/):
    shap_test_sample_results.json   -- full SHAP values for all 20 samples
    shap_test_sample_summary.csv    -- per-sample summary table
    shap_sample_NNNN.png            -- individual bar charts
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
    logger.info("Phase 3B — SHAP Explanation Runner")
    logger.info("=" * 60)

    from src.explain.shap_explainer import XLMRShapExplainer

    save_dir = PROJECT_ROOT / "results" / "shap"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model and build SHAP explainer
    # ------------------------------------------------------------------
    logger.info("\n[1/3] Loading XLM-R model and SHAP explainer...")
    explainer = XLMRShapExplainer()
    explainer.load_model()

    # ------------------------------------------------------------------
    # 2. Explain 20 stratified samples from the test set
    # ------------------------------------------------------------------
    logger.info("\n[2/3] Running SHAP on 20 stratified test samples...")
    results = explainer.explain_test_sample(
        n=20,
        save_dir=str(save_dir),
        max_evals=500,
        random_seed=42,
    )

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    logger.info("\n[3/3] Summary")
    logger.info("-" * 60)
    correct = sum(
        1 for r in results
        if r.get("predicted_label") == r.get("ground_truth_label")
    )
    logger.info(f"  Samples explained:   {len(results)}")
    logger.info(f"  Correct predictions: {correct}/{len(results)}")
    logger.info(f"  Results saved to:    {save_dir}")
    logger.info("")
    for i, r in enumerate(results):
        top_hof = max(zip(r["tokens"], r["shap_values_hof"]),
                      key=lambda x: abs(x[1]),
                      default=("?", 0.0))
        logger.info(
            f"  [{i:02d}] pred={r['predicted_label']} "
            f"gt={r.get('ground_truth_label','?')} "
            f"P(HOF)={r['predicted_prob_hof']:.3f}  "
            f"top_token={top_hof[0]!r} (shap={top_hof[1]:+.4f})"
        )

    logger.info("\n[OK] Phase 3B SHAP explanation complete.")
    logger.info(f"  JSON: {save_dir / 'shap_test_sample_results.json'}")
    logger.info(f"  CSV:  {save_dir / 'shap_test_sample_summary.csv'}")
    logger.info(f"  PNGs: {save_dir}/shap_sample_NNNN.png")


if __name__ == "__main__":
    main()
