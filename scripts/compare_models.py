"""
Script: Model Comparison & Results Reporting
==============================================
Aggregates all evaluation metrics from results/metrics/ and prints/saves
a comparison table.

Usage
-----
    python scripts/compare_models.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_project_root
from src.evaluation.metrics import build_comparison_table
from src.data.loaders import DEMO_MARKER

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    project_root = get_project_root()

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)

    df_comp = build_comparison_table(config=cfg)

    if df_comp.empty:
        print("\n  [WARNING] No metric results found in results/metrics/.")
        print("  Please train models first using:")
        print("      python scripts/train_baselines.py [--demo]")
        print("      python scripts/train_xlmr.py [--demo]\n")
        sys.exit(0)

    print("\n" + df_comp.to_string(index=False))

    # Check if any results are demo mode
    has_demo = df_comp["Demo Mode"].any() if "Demo Mode" in df_comp.columns else False
    if has_demo:
        print("\n" + "!" * 60)
        print(f"  [WARNING] NOTICE: Some or all results above are based on {DEMO_MARKER}.")
        print("  These outputs demonstrate SOFTWARE PIPELINE FUNCTIONALITY ONLY.")
        print("  DO NOT report these metrics as final research results.")
        print("!" * 60)

    # Save summary table
    out_dir = project_root / cfg["paths"]["results_metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison_summary.csv"
    df_comp.to_csv(out_path, index=False)
    print(f"\n  [OK] Model comparison table saved -> {out_path}\n")


if __name__ == "__main__":
    main()
