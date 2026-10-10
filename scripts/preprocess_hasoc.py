import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Paths
# ============================================================

INPUT_FILE = Path("data/raw/hasoc2021/HASOC_HINDI.csv")
OUTPUT_DIR = Path("data/processed/hasoc2021")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)


# ============================================================
# Select required columns
# ============================================================

df = df[["cleaned Tweets", "binary class"]].copy()

df.columns = ["text", "label"]


# ============================================================
# Remove missing values
# ============================================================

df = df.dropna(subset=["text", "label"])

# Remove empty texts
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]


# ============================================================
# Remove duplicate texts
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(subset=["text"], keep="first")

duplicates_removed = before_duplicates - len(df)

print("Duplicates removed:", duplicates_removed)


# ============================================================
# Normalize labels
# NOT = 0
# HOF = 1
# ============================================================

label_mapping = {
    "NOT": 0,
    "HOF": 1
}

df["label"] = df["label"].map(label_mapping)

# Make sure no unexpected labels exist
if df["label"].isna().any():
    print("\nUnexpected labels found:")
    print(df[df["label"].isna()])
    raise ValueError("Unexpected label found in HASOC dataset.")


df["label"] = df["label"].astype(int)


# ============================================================
# Shuffle dataset
# ============================================================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# Stratified split
#
# 80% Train
# 10% Validation
# 10% Test
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

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)


# ============================================================
# Verification
# ============================================================

for name, split in [
    ("train.csv", train_df),
    ("val.csv", val_df),
    ("test.csv", test_df)
]:

    print(f"\n===== {name} =====")
    print("Shape:", split.shape)

    print("\nLabel distribution:")
    print(split["label"].value_counts().sort_index())

    print("\nFirst 3 rows:")
    print(split.head(3).to_string(index=False))


print("\n================================")
print("HASOC preprocessing completed!")
print("================================")