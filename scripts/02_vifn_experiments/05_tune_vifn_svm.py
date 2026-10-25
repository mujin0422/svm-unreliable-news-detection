
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    make_scorer,
    precision_score,
)

SEED = 42

DATA_DIR = Path("data/processed/vifn_v1")
REPORT_DIR = Path("reports/vifn_v1")
MODEL_DIR = Path("models/vifn_v1")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ====================================
# 1. LOAD CLEAN TRAIN
# ====================================

train = pd.read_csv(DATA_DIR / "train.csv")

X_train = train["text"].fillna("").tolist()
y_train = train["label"].to_numpy()

assert len(train) == 924
assert train["text"].is_unique

# Audit trước đã xác nhận mỗi mẫu
# thuộc một nhóm độc lập
groups = np.arange(len(train))

print("=== DATASET ===")
print("Train samples:", len(train))
print("Label distribution:")
print(train["label"].value_counts())

# ====================================
# 2. MODEL PIPELINE
# ====================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
    )),
    ("svm", LinearSVC(
        random_state=SEED,
        max_iter=5000,
    )),
])

# ====================================
# 3. HYPERPARAMETER GRID
# ====================================

param_grid = {
    "svm__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "svm__class_weight": [None, "balanced"],
}

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "precision_1": make_scorer(
        precision_score,
        pos_label=1,
        zero_division=0,
    ),
    "recall_1": "recall",
    "f1_1": "f1",
}

# ====================================
# 4. GRID SEARCH
# ====================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="macro_f1",
    cv=cv,
    n_jobs=2,
    pre_dispatch=2,
    verbose=1,
    return_train_score=True,
    error_score="raise",
)

print("\n=== START GRID SEARCH ===")

grid.fit(
    X_train,
    y_train,
    groups=groups,
)

print("\n=== BEST PARAMETERS ===")
print(grid.best_params_)

print("\n=== BEST CV MACRO F1 ===")
print(round(grid.best_score_, 4))

# ====================================
# 5. SAVE RESULTS
# ====================================

cv_results = pd.DataFrame(grid.cv_results_)

columns = [
    "param_svm__C",
    "param_svm__class_weight",
    "mean_test_macro_f1",
    "std_test_macro_f1",
    "mean_test_precision_1",
    "mean_test_recall_1",
    "mean_test_f1_1",
    "mean_test_accuracy",
    "mean_train_macro_f1",
    "rank_test_macro_f1",
]

summary = cv_results[columns].sort_values(
    "rank_test_macro_f1"
)

summary.to_csv(
    REPORT_DIR / "svm_tuning_cv.csv",
    index=False,
)

print("\n=== CV COMPARISON ===")
print(summary.to_string(index=False))

# ====================================
# 6. SAVE BEST MODEL
# ====================================

# GridSearchCV tự refit mô hình tốt nhất
# trên toàn bộ 924 mẫu Train
best_model = grid.best_estimator_

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_word_bigram.joblib",
)

metadata = {
    "dataset": "ViFN V1 - Cleaned",
    "train_samples": len(train),
    "feature_extraction": "Word TF-IDF (1,2)",
    "classifier": "LinearSVC",
    "cv_folds": 5,
    "selection_metric": "Macro F1",
    "best_parameters": grid.best_params_,
    "best_cv_macro_f1": float(grid.best_score_),
    "test_evaluated": False,
}

with open(
    REPORT_DIR / "svm_tuning_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n=== COMPARISON WITH M03 ===")
print("M03 CV Macro F1: 0.8358")
print("Tuned CV Macro F1:",
      round(grid.best_score_, 4))

print("\nModel saved.")
print("ViFN SVM tuning completed!")
