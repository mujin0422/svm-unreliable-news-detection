
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import make_scorer, precision_score

SEED = 42

DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

X_train = train["text"].fillna("")
y_train = train["label"]

X_val = val["text"].fillna("")
y_val = val["label"]

# 2. Fixed feature extraction
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

# 3. Hyperparameter grid
param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__class_weight": [None, "balanced"],
}

# 4. Cross-validation
cv = StratifiedKFold(
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
        zero_division=0
    ),
    "recall_1": "recall",
    "f1_1": "f1",
}

# 5. GridSearch
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

print("=== START GRID SEARCH ===")
grid.fit(X_train, y_train)

print("\n=== BEST PARAMETERS ===")
print(grid.best_params_)

print("\nBest CV Macro F1:")
print(round(grid.best_score_, 4))

# 6. Save all CV results
cv_results = pd.DataFrame(grid.cv_results_)

columns = [
    "param_svm__C",
    "param_svm__class_weight",
    "mean_test_macro_f1",
    "std_test_macro_f1",
    "mean_test_f1_1",
    "mean_test_precision_1",
    "mean_test_recall_1",
    "mean_test_accuracy",
    "mean_train_macro_f1",
    "rank_test_macro_f1",
]

summary = cv_results[columns].sort_values(
    "rank_test_macro_f1"
)

summary.to_csv(
    REPORT_DIR / "experiment_svm_tuning_cv.csv",
    index=False,
)

print("\n=== CV COMPARISON ===")
print(summary.to_string(index=False))

# 7. Validate best CV model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_val)

print("\n=== VALIDATION RESULTS ===")

print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1],
    digits=4,
    zero_division=0,
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_val, y_pred, labels=[0, 1]
))

val_result = {
    "accuracy": accuracy_score(y_val, y_pred),
    "macro_f1": f1_score(
        y_val, y_pred, average="macro"
    ),
    "precision_1": precision_score(
        y_val, y_pred, zero_division=0
    ),
    "recall_1": recall_score(
        y_val, y_pred, zero_division=0
    ),
    "f1_1": f1_score(
        y_val, y_pred, zero_division=0
    ),
}

pd.DataFrame([val_result]).to_csv(
    REPORT_DIR / "experiment_svm_tuning_val.csv",
    index=False,
)

print("\n=== VALIDATION SUMMARY ===")
print(val_result)

print("\nPrevious best validation Macro F1: 0.7734")
print("New validation Macro F1:",
      round(val_result["macro_f1"], 4))

# 8. Save tuned model
joblib.dump(
    best_model,
    MODEL_DIR / "reintel_tuned_cv.joblib",
)

print("\nTuning completed!")
