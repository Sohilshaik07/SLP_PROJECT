"""
Dataset Loaders
===============
Generic, auto-detecting dataset loader designed to work with
multiple Indian code-mixed hate speech datasets.

DESIGN PRINCIPLES
-----------------
- Column names are NOT hard-coded for any dataset.
  The loader inspects the actual file first, then maps columns.
- Raw data is NEVER modified. All processed files go to data/processed/.
- Labels are preserved as `original_label`.
  A `normalized_label` is only created after an explicit, documented
  mapping is applied.
- Every loader reports exact statistics: shape, columns, label counts,
  missing values, duplicates.

DATASET STATUS (as of 2026-08-17)
----------------------------------
HASOC 2021    : Access requested. Awaiting credentials.
Bohra 2018    : Only tweet IDs available. Text not accessible.
HECM 2024     : GitHub repo returns 404. Inaccessible.
HOLD-Telugu   : Future dataset. Not yet verified.

Usage
-----
    from src.data.loaders import load_hasoc_dataset, inspect_dataframe
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import preprocess_dataframe, remove_duplicates
from src.utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_MARKER = "DEMO DATA — NOT PROJECT RESULTS"

# ---------------------------------------------------------------------------
# Core Inspection Utility
# ---------------------------------------------------------------------------

def inspect_dataframe(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Inspect a DataFrame and report key statistics.

    Parameters
    ----------
    df : pd.DataFrame
    dataset_name : str
        Human-readable name for logging.

    Returns
    -------
    dict
        Dictionary containing inspection results.
    """
    separator = "=" * 60
    report = {}

    report["dataset_name"] = dataset_name
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = list(df.columns)
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing = df.isnull().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing.items()}
    report["total_missing"] = int(sum(missing.values()))

    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Label distribution (if label column detected)
    label_cols = [c for c in df.columns if "label" in c.lower()]
    if label_cols:
        label_col = label_cols[0]
        vc = df[label_col].value_counts()
        report["label_column_detected"] = label_col
        report["label_distribution"] = vc.to_dict()
        report["label_percentages"] = (vc / len(df) * 100).round(2).to_dict()

    # Text length stats (if text column detected)
    text_cols = [c for c in df.columns if "text" in c.lower() or "post" in c.lower()
                 or "tweet" in c.lower() or "content" in c.lower()]
    if text_cols:
        text_col = text_cols[0]
        lengths = df[text_col].astype(str).str.len()
        report["text_column_detected"] = text_col
        report["text_length_stats"] = {
            "min":    int(lengths.min()),
            "max":    int(lengths.max()),
            "mean":   round(float(lengths.mean()), 2),
            "median": float(lengths.median()),
            "std":    round(float(lengths.std()), 2),
        }

    # Print report
    print(f"\n{separator}")
    print(f"  DATASET INSPECTION: {dataset_name}")
    print(separator)
    print(f"  Shape             : {report['shape']['rows']} rows × {report['shape']['columns']} columns")
    print(f"  Columns           : {report['columns']}")
    print(f"  Missing values    : {report['total_missing']}")
    print(f"  Duplicate rows    : {report['duplicate_rows']}")
    if "label_column_detected" in report:
        print(f"  Label column      : {report['label_column_detected']}")
        for lbl, cnt in report["label_distribution"].items():
            pct = report["label_percentages"].get(lbl, 0)
            print(f"    {lbl:20s}: {cnt:6d} ({pct:.1f}%)")
    if "text_column_detected" in report:
        s = report["text_length_stats"]
        print(f"  Text length (chars): min={s['min']}, max={s['max']}, "
              f"mean={s['mean']:.1f}, median={s['median']:.0f}")
    print(separator)

    return report


def _detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristically detect which column is text and which is label.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple of (text_col, label_col)
        Either may be None if not detectable.
    """
    # Common text column name patterns (case-insensitive)
    text_patterns = ["text", "post", "tweet", "content", "sentence",
                     "comment", "message", "description"]
    # Common label column name patterns
    label_patterns = ["label", "class", "category", "hate", "tag",
                      "annotation", "subtask_a", "subtask_b", "task_a", "task_b"]

    cols_lower = {c.lower(): c for c in df.columns}

    text_col = None
    for pat in text_patterns:
        matches = [original for lower, original in cols_lower.items() if pat in lower]
        if matches:
            text_col = matches[0]
            break

    label_col = None
    for pat in label_patterns:
        matches = [original for lower, original in cols_lower.items() if pat in lower]
        if matches:
            label_col = matches[0]
            break

    # Fallback: if only two columns, assume first is text and second is label
    if text_col is None and label_col is None and len(df.columns) == 2:
        text_col, label_col = df.columns[0], df.columns[1]

    return text_col, label_col


def _normalize_labels(
    df: pd.DataFrame,
    label_col: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    """
    Create a normalized binary label column (0/1).

    Preserves the original label as `original_label`.
    Creates `normalized_label` as integer (1 = positive/hate/HOF,
    0 = negative/normal/NOT).

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Name of the original label column.
    positive_label : str
        The value in label_col corresponding to the positive class.
    negative_label : str
        The value in label_col corresponding to the negative class.

    Returns
    -------
    pd.DataFrame
        With `original_label` and `normalized_label` columns added.
    """
    df = df.copy()

    # Preserve original
    df["original_label"] = df[label_col].astype(str).str.strip()

    unique_labels = df["original_label"].unique()
    logger.info(f"Unique labels found in '{label_col}': {unique_labels}")

    # Build mapping
    mapping = {}
    for lbl in unique_labels:
        if lbl == positive_label:
            mapping[lbl] = 1
        elif lbl == negative_label:
            mapping[lbl] = 0
        else:
            logger.warning(
                f"Unknown label '{lbl}' found in dataset. "
                f"Expected '{positive_label}' or '{negative_label}'. "
                f"Marking as NaN."
            )
            mapping[lbl] = None

    df["normalized_label"] = df["original_label"].map(mapping)
    unmapped = df["normalized_label"].isnull().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows have unmapped labels and will be dropped.")
        df = df.dropna(subset=["normalized_label"]).reset_index(drop=True)
        df["normalized_label"] = df["normalized_label"].astype(int)

    logger.info(
        f"Label normalization complete. "
        f"Mapping: {mapping} | "
        f"Positive (1): {(df['normalized_label'] == 1).sum()} | "
        f"Negative (0): {(df['normalized_label'] == 0).sum()}"
    )

    return df


# ---------------------------------------------------------------------------
# HOLD-Telugu Loader
# ---------------------------------------------------------------------------

def load_hold_telugu_dataset(
    raw_dir: Optional[str] = None,
    config: Optional[dict] = None,
    save_inspection: bool = True,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load the HOLD-Telugu (DravidianLangTech 2024) dataset.

    Loads the labeled training Excel file (`training_data_telugu-hate.xlsx`)
    and the unlabeled test Excel file (`telugu-hate-speech-test.xlsx`).

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing the raw HOLD-Telugu files.
    config : dict, optional
        Full config dict loaded from config.yaml.
    save_inspection : bool
        Save the inspection report to results/metrics/.

    Returns
    -------
    tuple of (train_df, unlabelled_test_df) or None
        train_df: Preprocessed DataFrame with original and normalized labels.
        unlabelled_test_df: Preprocessed DataFrame without labels (for inference only).
    """
    cfg = config or load_config()
    project_root = get_project_root()
    ht_cfg = cfg["dataset"].get("hold_telugu", {})

    if raw_dir is None:
        raw_dir = project_root / ht_cfg.get("raw_dir", "data/raw/hold_telugu")
    else:
        raw_dir = Path(raw_dir)

    train_file = raw_dir / ht_cfg.get("train_filename", "training_data_telugu-hate.xlsx")
    test_file  = raw_dir / ht_cfg.get("test_filename", "telugu-hate-speech-test.xlsx")

    if not train_file.exists():
        logger.error(f"HOLD-Telugu train file not found at: {train_file}")
        return None

    logger.info(f"Loading HOLD-Telugu train file: {train_file.name}")
    df_train_raw = pd.read_excel(train_file)

    text_col  = ht_cfg.get("text_column", "Comments")
    label_col = ht_cfg.get("label_column", "Label")
    pos_label = ht_cfg.get("positive_label", "hate")
    neg_label = ht_cfg.get("negative_label", "non-hate")

    report_train = inspect_dataframe(df_train_raw, dataset_name="HOLD-Telugu Train (Raw)")

    # Deduplicate & Clean
    df_train = remove_duplicates(df_train_raw, text_column=text_col)
    df_train = _normalize_labels(df_train, label_col, pos_label, neg_label)

    preprocess_cfg = cfg.get("preprocessing", {})
    df_train = preprocess_dataframe(df_train, text_col, config=preprocess_cfg)

    # Save processed labeled dataset
    processed_dir = project_root / cfg["paths"]["data_processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_train_path = processed_dir / "HOLD_TELUGU_processed_train.csv"
    df_train.to_csv(out_train_path, index=False, encoding="utf-8-sig")
    logger.info(f"Processed HOLD-Telugu labeled training dataset saved -> {out_train_path}")

    # Load unlabelled test file if present
    df_test_processed = None
    if test_file.exists():
        logger.info(f"Loading HOLD-Telugu unlabelled test file: {test_file.name}")
        df_test_raw = pd.read_excel(test_file, header=None)
        df_test_raw.columns = ["S.No", "Comments"]
        inspect_dataframe(df_test_raw, dataset_name="HOLD-Telugu Official Test (Unlabelled)")

        df_test_processed = preprocess_dataframe(df_test_raw, "Comments", config=preprocess_cfg)
        out_test_path = processed_dir / "HOLD_TELUGU_processed_unlabelled_test.csv"
        df_test_processed.to_csv(out_test_path, index=False, encoding="utf-8-sig")
        logger.info(f"Processed HOLD-Telugu unlabelled test file saved -> {out_test_path}")
    else:
        logger.warning(f"HOLD-Telugu test file not found at: {test_file}")

    if save_inspection:
        report_dir = project_root / cfg["paths"]["results_metrics"]
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "HOLD_TELUGU_inspection.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_train, f, indent=2)
        logger.info(f"Inspection report saved -> {report_path}")

    return df_train, df_test_processed


# ---------------------------------------------------------------------------
# HASOC 2021 Loader
# ---------------------------------------------------------------------------

def load_hasoc_dataset(
    raw_dir: Optional[str] = None,
    config: Optional[dict] = None,
    demo_mode: bool = False,
    save_inspection: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load the HASOC 2021 English-Hindi Code-Mix dataset.

    The function auto-detects column names on first run and does NOT
    assume any fixed schema. Column names are reported in the inspection
    output so they can be confirmed against the official documentation.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing the raw HASOC files.
        Defaults to data/raw/hasoc2021/ (from config).
    config : dict, optional
        Full config dict loaded from config.yaml.
        If None, loads automatically.
    demo_mode : bool
        If True, loads demo data from data/demo/ instead.
        All outputs are clearly marked as DEMO DATA.
    save_inspection : bool
        Save the inspection report to results/metrics/.

    Returns
    -------
    pd.DataFrame or None
        Loaded and inspected DataFrame, or None if data is not available.

    Notes
    -----
    Returns None if the dataset directory is empty or missing.
    This is the correct behaviour when awaiting dataset access —
    the pipeline can still be tested with demo_mode=True.
    """
    cfg = config or load_config()
    project_root = get_project_root()

    if demo_mode:
        return _load_demo_dataset(cfg)

    # Determine raw directory
    if raw_dir is None:
        raw_dir = project_root / cfg["paths"]["data_raw"] / "hasoc2021"
    else:
        raw_dir = Path(raw_dir)

    # Check if directory exists and has files
    if not raw_dir.exists():
        logger.error(f"HASOC 2021 directory not found: {raw_dir}")
        return None

    csv_files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.tsv")) + \
                list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.xlsx"))

    if not csv_files:
        logger.error(
            f"No data files found in {raw_dir}.\n"
            f"  → Place the official HASOC 2021 files there.\n"
            f"  → Run in demo mode with --demo for software testing."
        )
        return None

    # Load file(s)
    logger.info(f"Found {len(csv_files)} file(s) in {raw_dir}: "
                f"{[f.name for f in csv_files]}")

    dfs = []
    for fpath in csv_files:
        df_part = _load_single_file(fpath)
        if df_part is not None:
            df_part["_source_file"] = fpath.name
            dfs.append(df_part)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Auto-detect columns
    text_col_cfg = cfg["dataset"]["hasoc2021"].get("text_column")
    label_col_cfg = cfg["dataset"]["hasoc2021"].get("label_column")

    text_col, label_col = _detect_columns(df)
    if text_col_cfg:
        text_col = text_col_cfg
    if label_col_cfg:
        label_col = label_col_cfg

    if text_col is None or label_col is None:
        logger.error(
            f"Could not auto-detect text/label columns.\n"
            f"  Columns found: {list(df.columns)}\n"
            f"  Please set text_column and label_column in configs/config.yaml"
        )
        return None

    logger.info(f"Using text_column='{text_col}', label_column='{label_col}'")

    # Inspect
    report = inspect_dataframe(df, dataset_name="HASOC 2021 (Raw)")

    # Remove duplicates
    if cfg.get("preprocessing", {}).get("remove_exact_duplicates", True):
        df = remove_duplicates(df, text_column=text_col)

    # Drop rows with missing text
    missing_text = df[text_col].isnull() | (df[text_col].astype(str).str.strip() == "")
    if missing_text.sum() > 0:
        logger.warning(f"Dropping {missing_text.sum()} rows with missing text.")
        df = df[~missing_text].reset_index(drop=True)

    # Normalize labels
    pos_label = cfg["dataset"]["hasoc2021"].get("positive_label", "HOF")
    neg_label = cfg["dataset"]["hasoc2021"].get("negative_label", "NOT")
    df = _normalize_labels(df, label_col, pos_label, neg_label)

    # Preprocess text
    preprocess_cfg = cfg.get("preprocessing", {})
    df = preprocess_dataframe(df, text_col, config=preprocess_cfg)

    # Save processed version
    processed_dir = project_root / cfg["paths"]["data_processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "hasoc2021_processed.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Processed dataset saved to: {out_path}")

    # Save inspection report
    if save_inspection:
        report_dir = project_root / cfg["paths"]["results_metrics"]
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "hasoc2021_inspection.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Inspection report saved to: {report_path}")

    return df


def _load_demo_dataset(config: dict) -> pd.DataFrame:
    """
    Load the demo dataset for software pipeline testing.
    All outputs are clearly marked as DEMO DATA.
    """
    project_root = get_project_root()
    cfg_demo = config["dataset"]["demo"]
    demo_path = project_root / cfg_demo["raw_dir"] / cfg_demo["filename"]

    if not demo_path.exists():
        raise FileNotFoundError(
            f"Demo dataset not found: {demo_path}\n"
            f"Expected: data/demo/demo_dataset.csv"
        )

    df = pd.read_csv(demo_path, encoding="utf-8")

    print("\n" + "!" * 60)
    print(f"  {DEMO_MARKER}")
    print(f"  This data is ONLY for software pipeline testing.")
    print(f"  Do NOT use these results in any project report.")
    print("!" * 60)

    text_col  = cfg_demo["text_column"]
    label_col = cfg_demo["label_column"]
    pos_label = cfg_demo["positive_label"]
    neg_label = cfg_demo["negative_label"]

    inspect_dataframe(df, dataset_name=f"[DEMO] {DEMO_MARKER}")

    df = remove_duplicates(df, text_column=text_col)
    df = _normalize_labels(df, label_col, pos_label, neg_label)

    preprocess_cfg = config.get("preprocessing", {})
    df = preprocess_dataframe(df, text_col, config=preprocess_cfg)

    return df


def _load_single_file(fpath: Path) -> Optional[pd.DataFrame]:
    """Load a single CSV/TSV/Excel file robustly."""
    try:
        ext = fpath.suffix.lower()
        if ext == ".csv":
            # Try multiple encodings
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(fpath, encoding=enc)
                    logger.info(f"Loaded {fpath.name} ({enc}, {len(df)} rows)")
                    return df
                except UnicodeDecodeError:
                    continue
        elif ext in (".tsv", ".txt"):
            df = pd.read_csv(fpath, sep="\t", encoding="utf-8", on_bad_lines="skip")
            logger.info(f"Loaded {fpath.name} (TSV, {len(df)} rows)")
            return df
        elif ext == ".xlsx":
            df = pd.read_excel(fpath)
            logger.info(f"Loaded {fpath.name} (Excel, {len(df)} rows)")
            return df
        else:
            logger.warning(f"Unsupported file type: {fpath.suffix}")
            return None
    except Exception as e:
        logger.error(f"Failed to load {fpath.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Train / Val / Test Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    label_col: str = "normalized_label",
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.

    Uses stratified sampling to preserve class distribution.
    Uses random_state=42 for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Column to use for stratification.
    config : dict, optional
        Full config dict.

    Returns
    -------
    tuple of (train_df, val_df, test_df)
    """
    cfg = config or load_config()
    split_cfg = cfg.get("splitting", {})

    test_size  = split_cfg.get("test_size", 0.10)
    val_size   = split_cfg.get("val_size", 0.10)
    random_state = split_cfg.get("random_state", 42)
    stratify   = split_cfg.get("stratify", True)

    stratify_col = df[label_col] if stratify else None

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: train vs val
    # val_size relative to total so we adjust
    val_size_adjusted = val_size / (1.0 - test_size)
    stratify_col2 = train_val[label_col] if stratify else None

    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_col2,
    )

    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    logger.info(
        f"Split complete (seed={random_state}): "
        f"Train={len(train)} | Val={len(val)} | Test={len(test)}"
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        dist = split_df[label_col].value_counts().to_dict()
        logger.info(f"  {split_name} label distribution: {dist}")

    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    prefix: str = "hasoc2021",
    config: Optional[dict] = None,
    demo_mode: bool = False,
) -> None:
    """
    Save train/val/test splits to data/processed/.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    prefix : str
        Filename prefix (e.g., 'hasoc2021' or 'demo').
    config : dict, optional
    demo_mode : bool
        If True, marks filenames and content as DEMO DATA.
    """
    cfg = config or load_config()
    project_root = get_project_root()
    out_dir = project_root / cfg["paths"]["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if demo_mode:
        prefix = f"DEMO_{prefix}"
        print(f"\n  {DEMO_MARKER} — Saving demo splits.")

    for name, df in [("train", train), ("val", val), ("test", test)]:
        out_path = out_dir / f"{prefix}_{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {name} split ({len(df)} rows) → {out_path}")
