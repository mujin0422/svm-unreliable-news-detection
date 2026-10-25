
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Character-based representation for similarity
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=100000,
    dtype=np.float32,
)

# Fit vocabulary only on training data
vectorizer.fit(train["text"].fillna(""))

def check_similarity(ref_df, target_df, ref_name, target_name):
    X_ref = vectorizer.transform(
        ref_df["text"].fillna("")
    )

    X_target = vectorizer.transform(
        target_df["text"].fillna("")
    )

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=2,
    )

    nn.fit(X_ref)

    distances, indices = nn.kneighbors(X_target)

    similarities = np.clip(
        1 - distances[:, 0],
        0,
        1,
    )

    print(f"\n=== {target_name} vs {ref_name} ===")

    for threshold in [0.85, 0.90, 0.95, 0.99]:
        count = np.sum(similarities >= threshold)
        print(f"Similarity >= {threshold}: {count}")

    print("Maximum similarity:", similarities.max())

    return pd.DataFrame({
        "reference_split": ref_name,
        "target_split": target_name,
        "target_index": np.arange(len(target_df)),
        "nearest_reference_index": indices[:, 0],
        "similarity": similarities,
    })

results = []

results.append(
    check_similarity(train, val, "train", "val")
)

results.append(
    check_similarity(train, test, "train", "test")
)

results.append(
    check_similarity(val, test, "val", "test")
)

audit_df = pd.concat(results, ignore_index=True)

audit_df.to_csv(
    REPORT_DIR / "near_duplicate_audit.csv",
    index=False,
)

print("\nNear-duplicate audit completed!")
