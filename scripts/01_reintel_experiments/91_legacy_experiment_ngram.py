
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

experiments = {
    "word_unigram": (1, 1),
    "word_unigram_bigram": (1, 2),
}

results = []

for name, ngram_range in experiments.items():

    print(f"\n=== {name} ===")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            min_df=2,
            sublinear_tf=True,
        )),
        ("svm", LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    metrics = {
        "experiment": name,
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
        "n_features": len(
            model.named_steps["tfidf"]
            .get_feature_names_out()
        ),
    }

    results.append(metrics)

    print(classification_report(
        y_val, y_pred, digits=4,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ))

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "experiment_ngram.csv",
    index=False
)

print("\n=== EXPERIMENT COMPARISON ===")
print(results_df.to_string(index=False))

print("\nExperiment completed!")
