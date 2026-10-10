"""
Script: Exploratory Data Analysis (EDA)
=======================================
Generates text length statistics, class distribution plots, and EDA figures.
Saves figures to results/figures/.

Usage
-----
    python scripts/run_eda.py [--demo]
"""

import argparse
import sys
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, get_project_root
from src.data.loaders import DEMO_MARKER

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis Script")
    parser.add_argument("--demo", action="store_true", help="Run EDA on demo dataset")
    parser.add_argument(
        "--dataset", type=str, default="hold_telugu",
        choices=["hold_telugu", "hasoc2021", "demo"],
        help="Target dataset for EDA. Default is 'hold_telugu'."
    )
    args = parser.parse_args()

    cfg = load_config()
    project_root = get_project_root()
    fig_dir = project_root / cfg["paths"]["results_figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.demo or args.dataset == "demo":
        prefix = "DEMO_"
        dataset_label = "DEMO"
        file_prefix = "hasoc2021"
    elif args.dataset == "hold_telugu":
        prefix = "HOLD_TELUGU_"
        dataset_label = "HOLD-TELUGU"
        file_prefix = "HOLD_TELUGU"
    else:
        prefix = "HASOC2021_"
        dataset_label = "HASOC 2021"
        file_prefix = "hasoc2021"

    data_path = project_root / cfg["paths"]["data_processed"] / f"{file_prefix}_train.csv"

    if not data_path.exists():
        print(f"\n  [WARNING] Processed data file not found: {data_path}")
        print("  Please run `python scripts/download_data.py --dataset hold_telugu` first.\n")
        sys.exit(1)

    df = pd.read_csv(data_path)

    print("\n" + "=" * 60)
    print(f"  RUNNING EDA ({dataset_label})")
    print("=" * 60)
    if args.demo:
        print(f"  [WARNING] {DEMO_MARKER}\n")

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="original_label", data=df, ax=ax, palette="Set2")
    title_suffix = f"\n({DEMO_MARKER})" if args.demo else ""
    ax.set_title(f"Class Distribution{title_suffix}")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    dist_fig_path = fig_dir / f"{prefix}class_distribution.png"
    plt.savefig(dist_fig_path, dpi=150)
    plt.close()
    logger.info(f"Class distribution figure saved -> {dist_fig_path}")

    # 2. Text Length Distribution Plot
    df["char_length"] = df["cleaned_text"].astype(str).str.len()
    df["word_count"] = df["cleaned_text"].astype(str).str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["char_length"], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Character Length Distribution{title_suffix}")
    axes[0].set_xlabel("Character Count")

    sns.histplot(df["word_count"], kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"Word Count Distribution{title_suffix}")
    axes[1].set_xlabel("Word Count")

    plt.tight_layout()
    len_fig_path = fig_dir / f"{prefix}text_length_distribution.png"
    plt.savefig(len_fig_path, dpi=150)
    plt.close()
    logger.info(f"Text length distribution figure saved -> {len_fig_path}")

    print("\n  [OK] EDA complete. Figures saved in results/figures/\n")


if __name__ == "__main__":
    main()
