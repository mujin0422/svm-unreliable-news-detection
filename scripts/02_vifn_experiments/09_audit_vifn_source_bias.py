
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)

SEED = 42

TRAIN_PATH = Path("data/processed/vifn_v1/train.csv")
MODEL_PATH = Path(
    "models/vifn_v1/tuned_word_bigram.joblib"
)
OUTPUT_DIR = Path("reports/vifn_v1/source_bias")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MARKERS = [
    "vnexpress",
    "vtcnews",
    "socialnews",
    "hoithocuocsong",
]

# ====================================
# 1. LOAD CLEAN TRAIN AND MODEL
# ====================================

train = pd.read_csv(TRAIN_PATH)
model = joblib.load(MODEL_PATH)

texts = train["text"].fillna("").astype(str)
y = train["label"].to_numpy()

vectorizer = model.named_steps["tfidf"]
svm = model.named_steps["svm"]

features = vectorizer.get_feature_names_out()
weights = svm.coef_[0]

X = vectorizer.transform(texts)

assert len(train) == 924
assert list(svm.classes_) == [0, 1]

# Document frequency per class
real_counts = np.asarray(
    (X[y == 0] != 0).sum(axis=0)
).ravel()

fake_counts = np.asarray(
    (X[y == 1] != 0).sum(axis=0)
).ravel()

n_real = int((y == 0).sum())
n_fake = int((y == 1).sum())

# ====================================
# 2. GLOBAL FEATURE AUDIT
# ====================================

top_n = 30

real_indices = np.argsort(weights)[:top_n]
fake_indices = np.argsort(weights)[-top_n:][::-1]

records = []

for direction, indices in [
    ("Real", real_indices),
    ("Fake", fake_indices),
]:
    for idx in indices:
        records.append({
            "direction": direction,
            "feature": features[idx],
            "coefficient": float(weights[idx]),
            "real_count": int(real_counts[idx]),
            "fake_count": int(fake_counts[idx]),
            "real_rate": float(
                real_counts[idx] / n_real
            ),
            "fake_rate": float(
                fake_counts[idx] / n_fake
            ),
        })

feature_report = pd.DataFrame(records)

feature_report.to_csv(
    OUTPUT_DIR / "feature_class_distribution.csv",
    index=False,
)

# ====================================
# 3. CHECK SOURCE MARKERS
# ====================================

print("=== SOURCE MARKER AUDIT ===")

vocab = vectorizer.vocabulary_

for marker in SOURCE_MARKERS:
    idx = vocab.get(marker)

    if idx is None:
        print(f"\n{marker}: not in vocabulary")
        continue

    print(f"\nFeature: {marker}")
    print("Coefficient:", round(float(weights[idx]), 4))
    print(
        f"Real: {real_counts[idx]}/{n_real} "
        f"({real_counts[idx]/n_real:.2%})"
    )
    print(
        f"Fake: {fake_counts[idx]}/{n_fake} "
        f"({fake_counts[idx]/n_fake:.2%})"
    )

# ====================================
# 4. REMOVE SELECTED MARKERS
# ====================================

pattern = (
    r"(?i)\b(?:"
    + "|".join(SOURCE_MARKERS)
    + r")\b"
)

texts_masked = (
    texts.str.replace(pattern, " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\n=== SOURCE REMOVAL ===")
print(
    "Documents changed:",
    int((texts != texts_masked).sum())
)

# ====================================
# 5. SAME MODEL CONFIGURATION
# ====================================

def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            min_df=2,
            sublinear_tf=True,
        )),
        ("svm", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        )),
    ])


# All documents are independent after
# the previous near-duplicate audit
groups = np.arange(len(y))

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

folds = list(cv.split(texts, y, groups))

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "f1_fake": "f1",
}

# ====================================
# 6. ABLATION CROSS-VALIDATION
# ====================================

results = []

for name, input_texts in [
    ("original", texts),
    ("source_markers_removed", texts_masked),
]:
    print(f"\n=== CV: {name} ===")

    scores = cross_validate(
        create_model(),
        input_texts.tolist(),
        y,
        cv=folds,
        scoring=scoring,
        n_jobs=2,
        error_score="raise",
    )

    result = {
        "experiment": name,
        "macro_f1_mean": float(
            scores["test_macro_f1"].mean()
        ),
        "macro_f1_std": float(
            scores["test_macro_f1"].std()
        ),
        "f1_fake_mean": float(
            scores["test_f1_fake"].mean()
        ),
        "accuracy_mean": float(
            scores["test_accuracy"].mean()
        ),
    }

    results.append(result)
    print(result)

# ====================================
# 7. SAVE
# ====================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "source_ablation_cv.csv",
    index=False,
)

print("\n=== FINAL ABLATION COMPARISON ===")
print(comparison.to_string(index=False))

delta = (
    results[1]["macro_f1_mean"]
    - results[0]["macro_f1_mean"]
)

print("\nMacro F1 difference:", round(delta, 6))
print("Source bias analysis completed!")
