import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = Path("data/processed/unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset configuration
# ============================================================

DATASETS = {

    "Hindi": {
        "path": Path("data/processed/hasoc2021"),
        "files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv"
        },
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "HASOC2021"
    },

    "Telugu": {
    	"path": Path("data/processed"),
    	"files": {
             "train": "HOLD_TELUGU_train.csv",
             "val": "HOLD_TELUGU_val.csv",
             "test": "HOLD_TELUGU_test.csv"
       },
       "text_column": "text",
       "label_column": "label",
       "dataset_name": "HOLD"
    },

    "Tamil": {
        "path": Path("data/processed/Dravidian_Tamil"),
        "files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv"
        },
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "DravidianTamil"
    },

    "Malayalam": {
        "path": Path("data/processed/Dravidian_Malayalam"),
        "files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv"
        },
        "text_column": "text",
        "label_column": "label",
        "dataset_name": "DravidianMalayalam"
    }
}


# ============================================================
# Load dataset
# ============================================================

def load_dataset(language, config, split):

    file_path = config["path"] / config["files"][split]

    df = pd.read_csv(file_path)

    # Keep only required columns
    df = df[
        [
            config["text_column"],
            config["label_column"]
        ]
    ].copy()

    # Standardize column names
    df.columns = ["text", "label"]

    # Add language and dataset information
    df["language"] = language
    df["dataset"] = config["dataset_name"]

    return df


# ============================================================
# Load all datasets
# ============================================================

splits = {}

for split in ["train", "val", "test"]:

    print("\n" + "=" * 60)
    print(f"Preparing {split.upper()} split")
    print("=" * 60)

    all_data = []

    for language, config in DATASETS.items():

        df = load_dataset(language, config, split)

        print(
            f"{language:12s} -> "
            f"{len(df):6d} samples | "
            f"labels: {df['label'].value_counts().to_dict()}"
        )

        all_data.append(df)

    unified_df = pd.concat(
        all_data,
        ignore_index=True
    )

    splits[split] = unified_df


# ============================================================
# Remove cross-split exact-text leakage
#
# Priority:
# TEST > VAL > TRAIN
#
# Test examples are preserved.
# Validation examples are preserved.
# Training duplicates are removed.
# ============================================================

test_texts = set(splits["test"]["text"])

before = len(splits["val"])

splits["val"] = splits["val"][
    ~splits["val"]["text"].isin(test_texts)
].copy()

val_removed = before - len(splits["val"])


val_texts = set(splits["val"]["text"])

before = len(splits["train"])

splits["train"] = splits["train"][
    ~splits["train"]["text"].isin(test_texts)
    & ~splits["train"]["text"].isin(val_texts)
].copy()

train_removed = before - len(splits["train"])


print("\n" + "=" * 60)
print("CROSS-SPLIT DUPLICATE REMOVAL")
print("=" * 60)

print("Removed from validation:", val_removed)
print("Removed from training  :", train_removed)


# ============================================================
# Shuffle and save
# ============================================================

for split in ["train", "val", "test"]:

    unified_df = splits[split].copy()

    unified_df = unified_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    output_file = OUTPUT_DIR / f"{split}.csv"

    unified_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print(f"{split.upper()} FINAL")
    print("=" * 60)

    print("Shape:", unified_df.shape)

    print("\nLanguage distribution:")
    print(unified_df["language"].value_counts())

    print("\nLabel distribution:")
    print(unified_df["label"].value_counts())

    print("\nLanguage × Label:")
    print(
        pd.crosstab(
            unified_df["language"],
            unified_df["label"]
        )
    )

    print("\nSaved:", output_file)


# ============================================================
# Final verification
# ============================================================

print("\n" + "=" * 60)
print("FINAL CROSS-SPLIT LEAKAGE CHECK")
print("=" * 60)

train = pd.read_csv(OUTPUT_DIR / "train.csv")
val = pd.read_csv(OUTPUT_DIR / "val.csv")
test = pd.read_csv(OUTPUT_DIR / "test.csv")

train_texts = set(train["text"])
val_texts = set(val["text"])
test_texts = set(test["text"])

print("TRAIN ∩ VAL :", len(train_texts & val_texts))
print("TRAIN ∩ TEST:", len(train_texts & test_texts))
print("VAL ∩ TEST  :", len(val_texts & test_texts))

print("\nFinal dataset sizes:")
print("Train:", train.shape)
print("Val  :", val.shape)
print("Test :", test.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\n" + "=" * 60)
print("UNIFIED DATASET PREPARATION COMPLETED")
print("=" * 60)