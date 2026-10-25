
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
THRESHOLD = 0.95

INPUT_DIR = Path("data/processed/reintel")
OUTPUT_DIR = Path("data/processed/reintel_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Reconstruct cleaned dataset v1
frames = []

for name in ["train", "val", "test"]:
    frame = pd.read_csv(INPUT_DIR / f"{name}.csv")
    frames.append(frame[["text", "label"]])

df = pd.concat(frames, ignore_index=True)
df["text"] = df["text"].fillna("")

print("Total samples:", len(df))

# 2. Build unsupervised similarity representation
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=100000,
    dtype=np.float32,
)

X = vectorizer.fit_transform(df["text"])

# 3. Union-Find to construct duplicate groups
n = len(df)
parent = np.arange(n)
rank = np.zeros(n, dtype=np.int32)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        ra, rb = rb, ra

    parent[rb] = ra

    if rank[ra] == rank[rb]:
        rank[ra] += 1

# Compute pairwise similarity in batches
# without building a dense N x N matrix
pair_count = 0
batch_size = 64

for start in range(0, n, batch_size):
    stop = min(start + batch_size, n)

    similarities = (X[start:stop] @ X.T).tocoo()

    rows = start + similarities.row
    cols = similarities.col

    mask = (
        (rows < cols)
        & (similarities.data >= THRESHOLD)
    )

    for i, j in zip(rows[mask], cols[mask]):
        union(int(i), int(j))
        pair_count += 1

# 4. Assign group IDs
roots = np.array([find(i) for i in range(n)])
_, group_ids = np.unique(roots, return_inverse=True)

df["group_id"] = group_ids

print("Similarity pairs:", pair_count)
print("Duplicate groups:", df["group_id"].nunique())

# 5. Audit mixed-label groups
group_stats = df.groupby("group_id").agg(
    size=("label", "size"),
    n_labels=("label", "nunique"),
)

mixed_groups = group_stats[
    group_stats["n_labels"] > 1
]

print("Mixed-label groups:", len(mixed_groups))

df[df["group_id"].isin(mixed_groups.index)].to_csv(
    OUTPUT_DIR / "mixed_label_groups.csv",
    index=False,
)

print("Largest group size:", group_stats["size"].max())

# 6. Stratified group-aware splitting
# 5/7 train, 1/7 validation, 1/7 test
sgkf = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=SEED,
)

folds = list(sgkf.split(
    df["text"],
    df["label"],
    groups=df["group_id"],
))

test_idx = folds[0][1]
val_idx = folds[1][1]

train_idx = np.setdiff1d(
    np.arange(n),
    np.concatenate([test_idx, val_idx]),
)

splits = {
    "train": df.iloc[train_idx].copy(),
    "val": df.iloc[val_idx].copy(),
    "test": df.iloc[test_idx].copy(),
}

# 7. Verify group isolation
train_groups = set(splits["train"]["group_id"])
val_groups = set(splits["val"]["group_id"])
test_groups = set(splits["test"]["group_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# 8. Save splits
for name, split in splits.items():
    split = split.reset_index(drop=True)

    split.to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False,
    )

    print(f"\n=== {name.upper()} ===")
    print("Samples:", len(split))
    print("Ratio:", round(len(split) / n, 4))
    print("Label distribution:")
    print(split["label"].value_counts(normalize=True))
    print("Groups:", split["group_id"].nunique())

# Save full grouping audit
df.to_csv(
    OUTPUT_DIR / "group_assignments.csv",
    index=False,
)

print("\nGroup-aware splitting completed!")
