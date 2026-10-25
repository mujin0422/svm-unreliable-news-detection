
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    StratifiedGroupKFold,
    learning_curve,
)

SEED = 42

DATA_DIR = Path("data/processed/reintel_v2_reviewed")
REPORT_DIR = Path(
    "reports/reintel_v2_reviewed/learning_curve"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 1. LOAD TRAIN + VALIDATION
# =====================================

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

df = pd.concat(
    [train, val],
    ignore_index=True,
)

# Support common column names
text_col = next(
    (
        c for c in ["text", "post_message"]
        if c in df.columns
    ),
    None,
)

group_col = next(
    (
        c for c in [
            "group_id",
            "near_duplicate_group_id",
            "near_dup_group",
            "duplicate_group_id",
            "component_id",
        ]
        if c in df.columns
    ),
    None,
)

if text_col is None or group_col is None:
    raise SystemExit(
        "Required columns not found. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df[text_col].fillna("").astype(str).tolist()
y = df["label"].to_numpy()
groups = df[group_col].to_numpy()

assert len(df) == 3663
assert set(np.unique(y)) == {0, 1}
assert pd.notna(groups).all()

print("=== DATASET ===")
print("Samples:", len(df))
print("Independent groups:", len(np.unique(groups)))

print("\nLabel distribution:")
print(df["label"].value_counts())

# =====================================
# 2. FROZEN REINTEL MODEL
# =====================================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
    )),
    ("svm", LinearSVC(
        C=0.1,
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )),
])

# =====================================
# 3. GROUP-AWARE CV
# =====================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

folds = list(cv.split(X, y, groups))

# Verify that groups do not cross folds
for fold_idx, (tr_idx, va_idx) in enumerate(
    folds, start=1
):
    train_groups = set(groups[tr_idx])
    val_groups = set(groups[va_idx])

    assert train_groups.isdisjoint(val_groups)

    print(
        f"Fold {fold_idx}: "
        f"Train={len(tr_idx)}, "
        f"Validation={len(va_idx)}"
    )

# =====================================
# 4. LEARNING CURVE
# =====================================

sizes, train_scores, cv_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    cv=folds,
    scoring="f1_macro",
    shuffle=True,
    random_state=SEED,
    n_jobs=2,
    error_score="raise",
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)

cv_mean = cv_scores.mean(axis=1)
cv_std = cv_scores.std(axis=1)

# =====================================
# 5. SAVE RESULTS
# =====================================

results = pd.DataFrame({
    "train_size": sizes,
    "train_macro_f1_mean": train_mean,
    "train_macro_f1_std": train_std,
    "cv_macro_f1_mean": cv_mean,
    "cv_macro_f1_std": cv_std,
    "generalization_gap": train_mean - cv_mean,
})

results.to_csv(
    REPORT_DIR / "learning_curve.csv",
    index=False,
)

print("\n=== REINTEL LEARNING CURVE ===")
print(results.to_string(index=False))

# =====================================
# 6. PLOT
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    sizes,
    train_mean,
    marker="o",
    label="Train Macro F1",
)

plt.fill_between(
    sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
)

plt.plot(
    sizes,
    cv_mean,
    marker="o",
    label="CV Macro F1",
)

plt.fill_between(
    sizes,
    cv_mean - cv_std,
    cv_mean + cv_std,
    alpha=0.15,
)

plt.xlabel("Training samples per fold")
plt.ylabel("Macro F1")
plt.title("ReINTEL - TF-IDF + Linear SVM Learning Curve")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "learning_curve.png",
    dpi=200,
)
plt.close()

print("\nReINTEL Learning Curve completed!")
print("Final Test was not accessed.")
