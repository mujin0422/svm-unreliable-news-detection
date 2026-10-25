
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SEED = 42
DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

X_train = train["text"].fillna("")
y_train = train["label"]

X_val = val["text"].fillna("")
y_val = val["label"]


# Word TF-IDF
def word_tfidf():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
    )


# Character TF-IDF
def char_tfidf():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )


experiments = {
    "word_1_2": word_tfidf(),

    "char_wb_3_5": char_tfidf(),

    "word_char_wb": FeatureUnion([
        ("word", word_tfidf()),
        ("char", char_tfidf()),
    ]),
}

results = []
best_score = -1
best_name = None

for name, vectorizer in experiments.items():

    print(f"\n=== {name} ===")

    model = Pipeline([
        ("features", vectorizer),
        ("svm", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=SEED,
            max_iter=5000,
        )),
    ])

    start = perf_counter()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    elapsed = perf_counter() - start

    macro_f1 = f1_score(
        y_val, y_pred, average="macro"
    )

    result = {
        "experiment": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": macro_f1,
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
        "n_features": model.named_steps[
            "svm"
        ].coef_.shape[1],
        "time_seconds": round(elapsed, 2),
    }

    results.append(result)

    print(result)
    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ))

    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name

        joblib.dump(
            model,
            MODEL_DIR / "reintel_exp03_best.joblib"
        )

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "experiment_char_ngram.csv",
    index=False
)

print("\n=== FINAL COMPARISON ===")
print(results_df.to_string(index=False))

print("\nBest configuration:", best_name)
print("Best Macro F1:", round(best_score, 4))
