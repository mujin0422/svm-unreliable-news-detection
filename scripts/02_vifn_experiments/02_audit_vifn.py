
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/raw/vifn_v1")
REPORT_DIR = Path("reports/vifn_v1")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.85, 0.90, 0.95, 0.99]

# =====================================
# 1. LOAD DATA
# =====================================

train = pd.read_csv(DATA_DIR / "train_data.csv")

# Không đọc nhãn Test trong giai đoạn audit
test = pd.read_csv(
    DATA_DIR / "test_data.csv",
    usecols=["content"]
)

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()

train["text"] = train["content"].map(normalize_text)
test["text"] = test["content"].map(normalize_text)

print("=== DATASET ===")
print("Train:", len(train))
print("Test:", len(test))

# =====================================
# 2. EXACT DUPLICATES IN TRAIN
# =====================================

print("\n=== TRAIN DUPLICATES ===")

counts = train["text"].value_counts()

print("Duplicate rows:",
      train["text"].duplicated().sum())

print("Unique texts:", train["text"].nunique())

print("Repeated text groups:",
      (counts > 1).sum())

print("Largest duplicate group:", counts.max())

conflicts = train.groupby("text")["label"].nunique()

print("Conflicting exact groups:",
      (conflicts > 1).sum())

# =====================================
# 3. EXACT TRAIN-TEST OVERLAP
# =====================================

print("\n=== TRAIN-TEST EXACT OVERLAP ===")

overlap = test["text"].isin(set(train["text"]))

print("Test samples found in Train:",
      int(overlap.sum()))

print("Duplicates within Test:",
      int(test["text"].duplicated().sum()))

# =====================================
# 4. CHARACTER TF-IDF SIMILARITY
# =====================================

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=100000,
    dtype=np.float32,
)

# Fit trên Train, không fit trên Test
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

# =====================================
# 5. NEAR DUPLICATES WITHIN TRAIN
# =====================================

print("\n=== TRAIN NEAR DUPLICATES ===")

train_sim = cosine_similarity(X_train, X_train)
np.fill_diagonal(train_sim, 0)

for threshold in THRESHOLDS:
    count = (train_sim.max(axis=1) >= threshold).sum()
    print(f"Train rows with neighbor >= {threshold}: {count}")

# Các cặp gần trùng nhưng khác nhãn
rows, cols = np.where(
    np.triu(train_sim >= 0.95, k=1)
)

mixed_pairs = [
    (i, j, float(train_sim[i, j]))
    for i, j in zip(rows, cols)
    if train.iloc[i]["label"] != train.iloc[j]["label"]
]

print("Near-duplicate pairs with different labels:",
      len(mixed_pairs))

# =====================================
# 6. TRAIN-TEST NEAR DUPLICATES
# =====================================

print("\n=== TRAIN-TEST NEAR DUPLICATES ===")

similarity = cosine_similarity(X_test, X_train)

nearest_idx = similarity.argmax(axis=1)
nearest_sim = similarity.max(axis=1)

for threshold in THRESHOLDS:
    count = (nearest_sim >= threshold).sum()
    print(f"Test samples with similarity >= {threshold}: {count}")

print("Maximum similarity:", nearest_sim.max())

# =====================================
# 7. SAVE SUSPECT PAIRS
# =====================================

suspects = []

for test_idx, (train_idx, score) in enumerate(
    zip(nearest_idx, nearest_sim)
):
    if score >= 0.95:
        suspects.append({
            "test_index": test_idx,
            "train_index": int(train_idx),
            "similarity": float(score),
            "train_label": int(train.iloc[train_idx]["label"]),
            "train_text": train.iloc[train_idx]["text"],
            "test_text": test.iloc[test_idx]["text"],
        })

suspects_df = pd.DataFrame(suspects)

suspects_df.to_csv(
    REPORT_DIR / "train_test_near_duplicates.csv",
    index=False
)

print("\n=== TOP SUSPICIOUS PAIRS ===")

for row in sorted(
    suspects,
    key=lambda x: x["similarity"],
    reverse=True
)[:5]:
    print("\nSimilarity:", round(row["similarity"], 5))
    print("Train:", row["train_text"][:250])
    print("Test:", row["test_text"][:250])
    print("-" * 50)

print("\nViFN audit completed!")
