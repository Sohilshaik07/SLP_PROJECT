"""
Data Preprocessing Module
==========================
Provides text cleaning functions that preserve code-mixed
linguistic characteristics.

DESIGN PRINCIPLES
-----------------
- Code-mixing is the central linguistic feature being studied.
  Therefore we preserve Hindi words, English words, and
  mixed-script content.
- We do NOT: translate, stem, lemmatize, or remove code-mixed
  content.
- We DO: remove noise (URLs, mentions) and normalize whitespace.
- Every row retains both `raw_text` and `cleaned_text` so
  that the original data is never overwritten.

Usage
-----
    from src.data.preprocessing import clean_text, preprocess_dataframe
"""

import re
import logging
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual Cleaning Functions
# ---------------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs from text."""
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(pattern, " ", text)


def remove_mentions(text: str) -> str:
    """Remove Twitter @mentions."""
    return re.sub(r"@\w+", " ", text)


def process_hashtags(text: str, keep_text: bool = True) -> str:
    """
    Process hashtags.

    Parameters
    ----------
    text : str
    keep_text : bool
        If True, keeps the hashtag text without the # symbol.
        If False, removes entire hashtag.
    """
    if keep_text:
        return re.sub(r"#(\w+)", r"\1", text)
    else:
        return re.sub(r"#\w+", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_noise(text: str) -> str:
    """
    Remove obvious formatting noise.
    Preserves: letters (all scripts), digits, common punctuation,
    emojis, and spaces.
    Removes: repeated punctuation sequences (e.g., '....', '!!!!')
    replaced with single instance.
    """
    # Collapse repeated punctuation (e.g., "!!!" -> "!")
    text = re.sub(r"([!?.]){3,}", r"\1", text)
    # Remove invisible/control characters (keep printable + whitespace)
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))
    return text


# ---------------------------------------------------------------------------
# Main Cleaning Pipeline
# ---------------------------------------------------------------------------

def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    keep_hashtag_text: bool = True,
    normalize_ws: bool = True,
    preserve_emojis: bool = True,
) -> str:
    """
    Apply the minimal, code-mix-preserving cleaning pipeline to a
    single text string.

    Parameters
    ----------
    text : str
        Raw text to clean.
    remove_urls_flag : bool
        Remove URLs. Default True.
    remove_mentions_flag : bool
        Remove @mentions. Default True.
    keep_hashtag_text : bool
        Keep hashtag words without the # symbol. Default True.
    normalize_ws : bool
        Normalize whitespace. Default True.
    preserve_emojis : bool
        Keep emojis intact. Default True (does nothing to remove them).

    Returns
    -------
    str
        Cleaned text.

    Notes
    -----
    This function deliberately does NOT:
    - Translate Hindi to English
    - Remove Hindi/Devanagari characters
    - Stem or lemmatize tokens
    - Remove code-mixed content of any kind
    """
    if not isinstance(text, str):
        return ""

    if remove_urls_flag:
        text = remove_urls(text)

    if remove_mentions_flag:
        text = remove_mentions(text)

    text = process_hashtags(text, keep_text=keep_hashtag_text)
    text = remove_special_noise(text)

    if normalize_ws:
        text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------------------------
# DataFrame-Level Operations
# ---------------------------------------------------------------------------

def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply cleaning to a DataFrame.

    Creates a `cleaned_text` column alongside the original text column.
    The original column is NEVER overwritten.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least one text column.
    text_column : str
        Name of the column containing raw text.
    config : dict, optional
        preprocessing section of config.yaml.
        If None, uses defaults.

    Returns
    -------
    pd.DataFrame
        DataFrame with added `cleaned_text` column.
        Original text column is preserved unchanged as `raw_text`.
    """
    df = df.copy()

    if config is None:
        config = {}

    # Preserve raw text
    if "raw_text" not in df.columns:
        df["raw_text"] = df[text_column].astype(str)

    # Apply cleaning
    remove_urls_flag      = config.get("remove_urls", True)
    remove_mentions_flag  = config.get("remove_mentions", True)
    keep_hashtag_text     = config.get("preserve_hashtag_text", True)
    normalize_ws          = config.get("normalize_whitespace", True)
    preserve_emojis       = config.get("preserve_emojis", True)

    df["cleaned_text"] = df["raw_text"].apply(
        lambda t: clean_text(
            t,
            remove_urls_flag=remove_urls_flag,
            remove_mentions_flag=remove_mentions_flag,
            keep_hashtag_text=keep_hashtag_text,
            normalize_ws=normalize_ws,
            preserve_emojis=preserve_emojis,
        )
    )

    # Mark empty cleaned texts
    empty_mask = df["cleaned_text"].str.strip() == ""
    if empty_mask.sum() > 0:
        logger.warning(
            f"{empty_mask.sum()} rows have empty cleaned_text after preprocessing. "
            "These will be retained but flagged."
        )
        df.loc[empty_mask, "cleaned_text"] = "[EMPTY]"

    logger.info(
        f"Preprocessing complete. "
        f"Input rows: {len(df)} | "
        f"Non-empty cleaned: {(~empty_mask).sum()}"
    )

    return df


def remove_duplicates(
    df: pd.DataFrame,
    text_column: str = "raw_text",
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove exact duplicate rows based on raw text content.

    Parameters
    ----------
    df : pd.DataFrame
    text_column : str
        Column to check for duplicates. Default 'raw_text'.
    keep : str
        Which duplicate to keep: 'first' or 'last'. Default 'first'.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[text_column], keep=keep).reset_index(drop=True)
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows (kept '{keep}').")
    else:
        logger.info("No duplicate rows found.")
    return df
