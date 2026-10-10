import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw/Dravidian_Malayalam")
OUT_DIR = Path("data/processed/Dravidian_Malayalam")

OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [
    "mal_full_offensive_train.csv",
    "mal_full_offensive_dev.csv",
    "mal_full_offensive_test.csv",
]


# ============================================================
# 1. LOAD ALL RAW FILES
# ============================================================

dfs = []

for file in RAW_FILES:
    path = RAW_DIR / file

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label", "extra"],
        dtype=str
    )

    df = df[["text", "label"]]
    df["source_file"] = file

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Original combined shape:", df.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

df["text"] = df["text"].fillna("").astype(str).str.strip()
df["label"] = df["label"].fillna("").astype(str).str.strip()

before_empty = len(df)

df = df[df["text"] != ""].copy()

print("Empty texts removed:", before_empty - len(df))


# ============================================================
# 3. REMOVE NON-MALAYALAM
# ============================================================

before_not_malayalam = len(df)

df = df[df["label"] != "not-malayalam"].copy()

print(
    "not-malayalam samples removed:",
    before_not_malayalam - len(df)
)


# ============================================================
# 4. NORMALIZE LABELS
# ============================================================

df["label"] = df["label"].apply(
    lambda x: 0 if x == "Not_offensive" else 1
)

print("\nLabels after normalization:")
print(df["label"].value_counts())


# ============================================================
# 5. FIND CONFLICTING DUPLICATES
# ============================================================

label_counts = df.groupby("text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print(
    "\nConflicting unique texts:",
    len(conflicting_texts)
)


# Save conflicting examples for audit
conflicting_df = df[
    df["text"].isin(conflicting_texts)
].copy()

conflicting_df.to_csv(
    OUT_DIR / "removed_conflicting_samples.csv",
    index=False
)


# Remove all conflicting texts
df = df[
    ~df["text"].isin(conflicting_texts)
].copy()

print(
    "Samples removed because of conflicting labels:",
    len(conflicting_df)
)


# ============================================================
# 6. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

print(
    "Exact duplicate samples removed:",
    before_duplicates - len(df)
)


# ============================================================
# 7. FINAL CLEAN DATASET
# ============================================================

df = df[
    ["text", "label"]
].reset_index(drop=True)

print("\n===== CLEAN DATASET =====")
print("Shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 8. CREATE LEAKAGE-FREE SPLITS
# ============================================================

# 80% train
# 10% validation
# 10% test

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
# 9. SAVE SPLITS
# ============================================================

train_df.to_csv(
    OUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUT_DIR / "val.csv",
    index=False
)

test_df.to_csv(
    OUT_DIR / "test.csv",
    index=False
)


# ============================================================
# 10. SPLIT REPORT
# ============================================================

for name, split in [
    ("train.csv", train_df),
    ("val.csv", val_df),
    ("test.csv", test_df)
]:

    print(f"\n===== {name} =====")
    print("Shape:", split.shape)

    print("Labels:")
    print(split["label"].value_counts())

    print(
        "Duplicate texts:",
        split["text"].duplicated().sum()
    )


# ============================================================
# 11. CROSS-SPLIT LEAKAGE CHECK
# ============================================================

print("\n===== CROSS-SPLIT LEAKAGE CHECK =====")

print(
    "TRAIN ∩ VAL:",
    len(
        set(train_df["text"])
        & set(val_df["text"])
    )
)

print(
    "TRAIN ∩ TEST:",
    len(
        set(train_df["text"])
        & set(test_df["text"])
    )
)

print(
    "VAL ∩ TEST:",
    len(
        set(val_df["text"])
        & set(test_df["text"])
    )
)


print("\nMalayalam preprocessing completed successfully!")