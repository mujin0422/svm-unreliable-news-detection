
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from underthesea import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)

# =====================================
# CONFIG
# =====================================

SEED = 42

DATA_DIR = Path("data/processed/reintel_v2_reviewed")
REPORT_DIR = Path("reports/reintel_v2_reviewed")
MODEL_DIR = Path("models/reintel_v2_reviewed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

X_train = train["text"].fillna("").tolist()
y_train = train["label"]

X_val = val["text"].fillna("").tolist()
y_val = val["label"]

groups = train["group_id"]

assert set(train["group_id"]).isdisjoint(
    set(val["group_id"])
)

print("=== DATASET ===")
print("Train:", len(train))
print("Validation:", len(val))
print("Train groups:", groups.nunique())

# =====================================
# FEATURE EXTRACTORS
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


def combined_tfidf():
    return FeatureUnion([
        ("word", word_tfidf()),
        ("char", char_tfidf()),
    ])


def build_pipeline(features, class_weight="balanced", C=1.0):
    return Pipeline([
        ("features", features),
        ("svm", LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=SEED,
            max_iter=5000,
        )),
    ])


# =====================================
# VIETNAMESE SEGMENTATION
# =====================================

print("\nSegmenting Vietnamese text...")

X_train_segmented = [
    word_tokenize(
        text,
        format="text",
        use_token_normalize=False,
    )
    for text in X_train
]

X_val_segmented = [
    word_tokenize(
        text,
        format="text",
        use_token_normalize=False,
    )
    for text in X_val
]

# =====================================
# FIXED EXPERIMENTS
# =====================================

experiments = {
    "M01_word_unigram": {
        "features": lambda: word_tfidf((1, 1)),
        "weight": None,
        "segmented": False,
    },
    "M02_word_unigram_balanced": {
        "features": lambda: word_tfidf((1, 1)),
        "weight": "balanced",
        "segmented": False,
    },
    "M03_word_bigram": {
        "features": lambda: word_tfidf((1, 2)),
        "weight": "balanced",
        "segmented": False,
    },
    "M04_word_segmented": {
        "features": lambda: word_tfidf((1, 2)),
        "weight": "balanced",
        "segmented": True,
    },
    "M05_char_wb": {
        "features": char_tfidf,
        "weight": "balanced",
        "segmented": False,
    },
    "M06_word_char": {
        "features": combined_tfidf,
        "weight": "balanced",
        "segmented": False,
    },
}

results = []

for name, config in experiments.items():

    print(f"\n=== {name} ===")

    model = build_pipeline(
        features=config["features"](),
        class_weight=config["weight"],
    )

    if config["segmented"]:
        tr_text = X_train_segmented
        va_text = X_val_segmented
    else:
        tr_text = X_train
        va_text = X_val

    start = perf_counter()

    model.fit(tr_text, y_train)
    predictions = model.predict(va_text)

    elapsed = perf_counter() - start

    cm = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
        "experiment": name,
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val, predictions, average="macro"
        ),
        "precision_1": precision_score(
            y_val, predictions, pos_label=1,
            zero_division=0
        ),
        "recall_1": recall_score(
            y_val, predictions, pos_label=1,
            zero_division=0
        ),
        "f1_1": f1_score(
            y_val, predictions, pos_label=1,
            zero_division=0
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n_features": model.named_steps[
            "svm"
        ].coef_.shape[1],
        "time_seconds": round(elapsed, 2),
    }

    results.append(result)

    print(result)

# =====================================
# SAVE FIXED EXPERIMENTS
# =====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "fixed_experiments.csv",
    index=False,
)

print("\n=== FIXED EXPERIMENT SUMMARY ===")
print(
    results_df.sort_values(
        "macro_f1", ascending=False
    ).to_string(index=False)
)

# =====================================
# GROUP-AWARE SVM TUNING
# =====================================

print("\n=== GROUP-AWARE GRID SEARCH ===")

pipeline = build_pipeline(
    features=word_tfidf((1, 2))
)

param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
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

grid.fit(
    X_train,
    y_train,
    groups=groups,
)

print("\n=== BEST CV PARAMETERS ===")
print(grid.best_params_)

print("Best CV Macro F1:", round(grid.best_score_, 4))

# =====================================
# SAVE CROSS-VALIDATION RESULTS
# =====================================

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

cv_summary = cv_results[columns].sort_values(
    "rank_test_macro_f1"
)

cv_summary.to_csv(
    REPORT_DIR / "svm_tuning_cv.csv",
    index=False,
)

print("\n=== CROSS-VALIDATION RESULTS ===")
print(cv_summary.to_string(index=False))

# =====================================
# VALIDATE TUNED MODEL
# =====================================

best_model = grid.best_estimator_

y_pred = best_model.predict(X_val)

print("\n=== TUNED MODEL VALIDATION ===")

print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1],
    target_names=["Reliable", "Unreliable"],
    digits=4,
    zero_division=0,
))

print(confusion_matrix(
    y_val, y_pred, labels=[0, 1]
))

tuned_result = {
    "accuracy": accuracy_score(y_val, y_pred),
    "macro_f1": f1_score(
        y_val, y_pred, average="macro"
    ),
    "precision_1": precision_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    ),
    "recall_1": recall_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    ),
    "f1_1": f1_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    ),
}

pd.DataFrame([tuned_result]).to_csv(
    REPORT_DIR / "tuned_validation.csv",
    index=False,
)

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_word_bigram.joblib",
)

print("\n=== TUNED VALIDATION SUMMARY ===")
print(tuned_result)

print("\nAll experiments completed!")
