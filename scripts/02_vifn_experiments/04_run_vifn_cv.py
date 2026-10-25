
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
from underthesea import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

SEED = 42
THRESHOLD = 0.95

DATA_DIR = Path("data/processed/vifn_v1")
RAW_DIR = Path("data/raw/vifn_v1")
REPORT_DIR = Path("reports/vifn_v1")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 1. LOAD DATA
# =====================================

train = pd.read_csv(DATA_DIR / "train.csv")
X = train["text"].fillna("").tolist()
y = train["label"].to_numpy()

print("Train samples:", len(X))
print("Label distribution:")
print(train["label"].value_counts())

assert len(train) == 924
assert train["text"].is_unique

# =====================================
# 2. NEAR-DUPLICATE CHECK
# =====================================

similarity_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=100000,
    dtype=np.float32,
)

X_sim = similarity_vectorizer.fit_transform(X)

sim = cosine_similarity(X_sim, X_sim)
np.fill_diagonal(sim, 0)

rows, cols = np.where(
    np.triu(sim >= THRESHOLD, k=1)
)

print("\n=== TRAIN NEAR-DUPLICATES ===")
print("Near-duplicate pairs:", len(rows))

# Union-Find grouping
parent = np.arange(len(X))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for i, j in zip(rows, cols):
    union(int(i), int(j))

roots = np.array([find(i) for i in range(len(X))])
_, groups = np.unique(roots, return_inverse=True)

print("Independent groups:", len(np.unique(groups)))

group_df = pd.DataFrame({
    "group_id": groups,
    "label": y,
})

conflicts = group_df.groupby(
    "group_id"
)["label"].nunique()

n_conflicts = int((conflicts > 1).sum())

print("Mixed-label groups:", n_conflicts)

if n_conflicts > 0:
    raise SystemExit(
        "Mixed-label groups detected. "
        "Review them before training."
    )

# =====================================
# 3. TRAIN-TEST LEAKAGE CHECK
# =====================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()

# Only Test text; never access Test labels
test_texts = pd.read_csv(
    RAW_DIR / "test_data.csv",
    usecols=["content"],
)["content"].map(normalize_text)

X_test_sim = similarity_vectorizer.transform(test_texts)

test_similarity = cosine_similarity(
    X_test_sim, X_sim
)

nearest = test_similarity.max(axis=1)

print("\n=== TRAIN-TEST LEAKAGE CHECK ===")

for threshold in [0.90, 0.95, 0.99]:
    print(
        f"Test samples >= {threshold}:",
        int((nearest >= threshold).sum())
    )

if (nearest >= THRESHOLD).any():
    raise SystemExit(
        "Remaining Train-Test near-duplicates detected. "
        "Review overlap before running CV."
    )

print("Near-duplicate leakage check passed.")

# =====================================
# 4. FEATURE EXTRACTORS
# =====================================

def word_tfidf(ngram=(1, 2)):
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram,
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
    )

def char_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

def build_model(features, class_weight):
    return Pipeline([
        ("features", features),
        ("svm", LinearSVC(
            C=1.0,
            class_weight=class_weight,
            random_state=SEED,
            max_iter=5000,
        )),
    ])

# =====================================
# 5. SEGMENTATION
# =====================================

print("\nSegmenting Vietnamese text...")

X_segmented = [
    word_tokenize(
        text,
        format="text",
        use_token_normalize=False,
    )
    for text in X
]

# =====================================
# 6. EXPERIMENT CONFIGURATIONS
# =====================================

experiments = {
    "M01_word_unigram": (
        lambda: word_tfidf((1, 1)), None, False
    ),
    "M02_unigram_balanced": (
        lambda: word_tfidf((1, 1)), "balanced", False
    ),
    "M03_word_bigram": (
        lambda: word_tfidf((1, 2)), "balanced", False
    ),
    "M04_word_segmented": (
        lambda: word_tfidf((1, 2)), "balanced", True
    ),
    "M05_char_wb": (
        char_tfidf, "balanced", False
    ),
    "M06_word_char": (
        lambda: FeatureUnion([
            ("word", word_tfidf((1, 2))),
            ("char", char_tfidf()),
        ]),
        "balanced",
        False,
    ),
}

# =====================================
# 7. GROUP-AWARE CROSS-VALIDATION
# =====================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

folds = list(cv.split(X, y, groups))

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "precision_1": make_scorer(
        precision_score, pos_label=1, zero_division=0
    ),
    "recall_1": "recall",
    "f1_1": "f1",
}

results = []

for name, (feature_factory, weight, segmented) in (
    experiments.items()
):
    print(f"\n=== {name} ===")

    model = build_model(
        feature_factory(),
        weight,
    )

    texts = X_segmented if segmented else X

    scores = cross_validate(
        model,
        texts,
        y,
        cv=folds,
        scoring=scoring,
        n_jobs=2,
        error_score="raise",
    )

    result = {"experiment": name}

    for metric in scoring:
        values = scores[f"test_{metric}"]
        result[f"mean_{metric}"] = values.mean()
        result[f"std_{metric}"] = values.std()

    results.append(result)

    print(result)

# =====================================
# 8. SAVE RESULTS
# =====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_macro_f1",
    ascending=False,
)

results_df.to_csv(
    REPORT_DIR / "fixed_experiments_cv.csv",
    index=False,
)

print("\n=== FINAL CV COMPARISON ===")

print(results_df.to_string(index=False))

print("\nBest feature configuration:")
print(results_df.iloc[0]["experiment"])

print("\nViFN CV experiments completed!")
