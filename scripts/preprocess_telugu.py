import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path(
    "data/raw/hold_telugu/training_data_telugu-hate.xlsx"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_excel(INPUT_FILE)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# Select required columns
# ============================================================

df = df[
    ["Comments", "Label"]
].copy()

df.columns = [
    "text",
    "original_label"
]


# ============================================================
# Remove missing values
# ============================================================

df = df.dropna(
    subset=["text", "original_label"]
).copy()


# ============================================================
# Clean text
# ============================================================

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)

df = df[
    df["text"] != ""
].copy()

print("After empty-text removal:", df.shape)


# ============================================================
# Normalize labels
#
# non-hate = 0
# hate     = 1
# ============================================================

label_mapping = {
    "non-hate": 0,
    "hate": 1
}

df["label"] = df["original_label"].map(
    label_mapping
)


# ============================================================
# Check for unexpected labels
# ============================================================

if df["label"].isna().any():

    print("\nUnexpected labels found:")

    print(
        df[
            df["label"].isna()
        ]["original_label"].value_counts()
    )

    raise ValueError(
        "Unexpected Telugu label found."
    )


df["label"] = df["label"].astype(int)


# ============================================================
# Detect conflicting duplicate texts
# ============================================================

label_counts = (
    df.groupby("text")["label"]
    .nunique()
)

conflicting_texts = set(
    label_counts[
        label_counts > 1
    ].index
)

print(
    "Conflicting unique texts:",
    len(conflicting_texts)
)


# ============================================================
# Remove conflicting texts
# ============================================================

if conflicting_texts:

    print("\nConflicting examples:")

    print(
        df[
            df["text"].isin(conflicting_texts)
        ][
            ["text", "original_label", "label"]
        ].to_string(index=False)
    )

    df = df[
        ~df["text"].isin(conflicting_texts)
    ].copy()

    print(
        "Removed conflicting samples:",
        len(conflicting_texts)
    )


# ============================================================
# Remove exact duplicate texts
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

duplicates_removed = (
    before_duplicates - len(df)
)

print(
    "Exact duplicate rows removed:",
    duplicates_removed
)


# ============================================================
# Keep final columns
# ============================================================

df = df[
    ["text", "label"]
].copy()


# ============================================================
# Shuffle
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# Final clean dataset
# ============================================================

print("\n===== CLEAN TELUGU DATASET =====")

print(
    "Shape:",
    df.shape
)

print("\nLabel distribution:")

print(
    df["label"].value_counts()
    .sort_index()
)


# ============================================================
# Train / validation / test split
#
# 80% / 10% / 10%
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ============================================================
# Save
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "HOLD_TELUGU_train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "HOLD_TELUGU_val.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "HOLD_TELUGU_test.csv",
    index=False
)


# ============================================================
# Verification
# ============================================================

for name, split in [
    ("HOLD_TELUGU_train.csv", train_df),
    ("HOLD_TELUGU_val.csv", val_df),
    ("HOLD_TELUGU_test.csv", test_df)
]:

    print("\n========================================")
    print(name)
    print("========================================")

    print(
        "Shape:",
        split.shape
    )

    print("\nLabels:")

    print(
        split["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nDuplicate texts:",
        split["text"].duplicated().sum()
    )


# ============================================================
# Cross-split leakage check
# ============================================================

train_texts = set(
    train_df["text"]
)

val_texts = set(
    val_df["text"]
)

test_texts = set(
    test_df["text"]
)

print("\n========================================")
print("CROSS-SPLIT LEAKAGE CHECK")
print("========================================")

print(
    "TRAIN ∩ VAL:",
    len(train_texts & val_texts)
)

print(
    "TRAIN ∩ TEST:",
    len(train_texts & test_texts)
)

print(
    "VAL ∩ TEST:",
    len(val_texts & test_texts)
)


print("\n========================================")
print("TELUGU PREPROCESSING COMPLETED")
print("========================================")