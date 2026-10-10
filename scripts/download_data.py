"""
Script: Data Preparation / Ingestion & Verification
===================================================
Prepares and inspects dataset files.
Supports both actual dataset loading (from data/raw/hasoc2021/) and
demo mode (from data/demo/demo_dataset.csv).

Usage
-----
    # Run with HASOC 2021 dataset (when files are placed in data/raw/hasoc2021/)
    python scripts/download_data.py

    # Run in DEMO mode for software pipeline verification
    python scripts/download_data.py --demo
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.data.loaders import load_hasoc_dataset, load_hold_telugu_dataset, split_dataset, save_splits, DEMO_MARKER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Data Ingestion & Verification Script")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run using demo dataset (for software pipeline testing only)."
    )
    parser.add_argument(
        "--dataset", type=str, default="hold_telugu",
        choices=["hold_telugu", "hasoc2021", "demo"],
        help="Target dataset to process. Default is 'hold_telugu'."
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.demo or args.dataset == "demo":
        print("\n" + "=" * 60)
        print(f"  RUNNING IN DEMO MODE: {DEMO_MARKER}")
        print("=" * 60)
        df = load_hasoc_dataset(config=cfg, demo_mode=True)
        if df is not None:
            train, val, test = split_dataset(df, config=cfg)
            save_splits(train, val, test, prefix="hasoc2021", config=cfg, demo_mode=True)
            print(f"\n  [OK] DEMO data pipeline verified successfully.")
            print(f"  [OK] Processed splits saved in data/processed/\n")

    elif args.dataset == "hold_telugu":
        print("\n" + "=" * 60)
        print("  RUNNING HOLD-TELUGU DATASET INGESTION")
        print("=" * 60)
        res = load_hold_telugu_dataset(config=cfg)
        if res is None:
            print("\n  [WARNING] HOLD-Telugu dataset files not found in data/raw/hold_telugu/.")
            sys.exit(1)

        df_train, df_test_unlabelled = res
        train, val, test = split_dataset(df_train, config=cfg)
        save_splits(train, val, test, prefix="HOLD_TELUGU", config=cfg, demo_mode=False)
        print("\n  [OK] HOLD-Telugu dataset processed and 80/10/10 internal evaluation splits saved in data/processed/")
        if df_test_unlabelled is not None:
            print("  [OK] Official unlabelled test set processed and saved for inference.\n")

    elif args.dataset == "hasoc2021":
        print("\n" + "=" * 60)
        print("  RUNNING HASOC 2021 DATASET INGESTION")
        print("=" * 60)
        df = load_hasoc_dataset(config=cfg, demo_mode=False)
        if df is None:
            print("\n  [WARNING] No dataset found in data/raw/hasoc2021/.")
            print("  Please request and place HASOC 2021 dataset files into data/raw/hasoc2021/.")
            print("  To test the software pipeline using demo data, run:")
            print("      python scripts/download_data.py --demo\n")
            sys.exit(0)

        train, val, test = split_dataset(df, config=cfg)
        save_splits(train, val, test, prefix="hasoc2021", config=cfg, demo_mode=False)
        print("\n  [OK] HASOC 2021 dataset processed and splits saved in data/processed/\n")


if __name__ == "__main__":
    main()
