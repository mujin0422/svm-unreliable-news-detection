
from pathlib import Path
import unicodedata
import json

import pandas as pd

SEED = 42

RAW_DIR = Path("data/raw/vifn_v1")
OUTPUT_DIR = Path("data/processed/vifn_v1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================
# 1. LOAD DATA
# ======================================

train = pd.read_csv(RAW_DIR / "train_data.csv")

# Chỉ đọc văn bản Test, không đọc nhãn
test_text = pd.read_csv(
    RAW_DIR / "test_data.csv",
    usecols=["content"]
)

assert train["content"].notna().all()
assert test_text["content"].notna().all()

# ======================================
# 2. NORMALIZE TEXT
# ======================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()

train["text"] = train["content"].map(normalize_text)
test_text["text"] = test_text["content"].map(
    normalize_text
)

assert (train["text"] != "").all()
assert (test_text["text"] != "").all()

# ======================================
# 3. REMOVE TRAIN DUPLICATES
# ======================================

original_count = len(train)

train = train.drop_duplicates(
    subset=["text"],
    keep="first"
).copy()

after_dedup = len(train)

# ======================================
# 4. REMOVE TRAIN-TEST OVERLAP
# ======================================

test_set = set(test_text["text"])

overlap_mask = train["text"].isin(test_set)

overlap_unique = int(overlap_mask.sum())

train_clean = train.loc[~overlap_mask].copy()

# ======================================
# 5. FINAL CHECKS
# ======================================

assert train_clean["text"].is_unique

assert set(train_clean["text"]).isdisjoint(
    test_set
)

assert train_clean["label"].isin([0, 1]).all()

train_clean = train_clean[
    ["text", "label"]
].reset_index(drop=True)

# ======================================
# 6. SAVE CLEAN TRAIN
# ======================================

train_clean.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

# Lưu metadata của quá trình làm sạch
audit = {
    "dataset": "ViFN V1",
    "original_train": original_count,
    "after_dedup": after_dedup,
    "removed_train_duplicates": (
        original_count - after_dedup
    ),
    "removed_test_overlap_unique": overlap_unique,
    "final_train": len(train_clean),
    "original_test": len(test_text),
    "test_internal_duplicates": int(
        test_text["text"].duplicated().sum()
    ),
    "test_modified": False,
    "seed": SEED,
}

with open(
    OUTPUT_DIR / "cleaning_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        audit,
        f,
        ensure_ascii=False,
        indent=2
    )

# ======================================
# 7. REPORT
# ======================================

print("=== VIFN DATA PREPARATION ===")

print("Original train:", original_count)
print("After dedup:", after_dedup)

print(
    "Unique train texts overlapping Test:",
    overlap_unique
)

print("Final clean train:", len(train_clean))
print("Original test:", len(test_text))

print("\n=== LABEL DISTRIBUTION ===")
print(train_clean["label"].value_counts())

print("\n=== LABEL PROPORTION ===")
print(
    train_clean["label"].value_counts(
        normalize=True
    )
)

print("\nTrain-Test exact overlap: 0")
print("Test labels not accessed")
print("ViFN preparation completed!")
