
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    StratifiedGroupKFold,
    learning_curve,
)

SEED = 42

DATA_PATH = Path("data/processed/vifn_v1/train.csv")
REPORT_DIR = Path("reports/vifn_v1/learning_curve")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 1. LOAD CLEAN TRAIN
# =====================================

df = pd.read_csv(DATA_PATH)

X = df["text"].fillna("").tolist()
y = df["label"].to_numpy()

assert len(df) == 924
assert df["text"].is_unique

groups = np.arange(len(df))

# =====================================
# 2. FROZEN MODEL CONFIGURATION
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
        C=1.0,
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )),
])

# =====================================
# 3. CROSS-VALIDATION
# =====================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

folds = list(cv.split(X, y, groups))

# =====================================
# 4. LEARNING CURVE
# =====================================

train_sizes, train_scores, val_scores = learning_curve(
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

val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# =====================================
# 5. SAVE RESULTS
# =====================================

results = pd.DataFrame({
    "train_size": train_sizes,
    "train_macro_f1_mean": train_mean,
    "train_macro_f1_std": train_std,
    "cv_macro_f1_mean": val_mean,
    "cv_macro_f1_std": val_std,
    "generalization_gap": train_mean - val_mean,
})

results.to_csv(
    REPORT_DIR / "learning_curve.csv",
    index=False,
)

print("=== VIFN LEARNING CURVE ===")
print(results.to_string(index=False))

# =====================================
# 6. PLOT
# =====================================

plt.figure(figsize=(9, 5))

plt.plot(
    train_sizes, train_mean,
    marker="o", label="Train Macro F1"
)

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15
)

plt.plot(
    train_sizes, val_mean,
    marker="o", label="CV Macro F1"
)

plt.fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.15
)

plt.xlabel("Training samples per fold")
plt.ylabel("Macro F1")
plt.title("ViFN - TF-IDF + Linear SVM Learning Curve")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "learning_curve.png",
    dpi=200
)

plt.close()

print("\nLearning Curve Analysis completed!")
print("No final Test data accessed.")
