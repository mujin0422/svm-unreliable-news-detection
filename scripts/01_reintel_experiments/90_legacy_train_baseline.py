
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==============================
# CONFIG
# ==============================

SEED = 42
DATA_DIR = Path("data/processed/reintel")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "val.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["label"]

X_val = val_df["text"].fillna("")
y_val = val_df["label"]

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

# ==============================
# EXPERIMENT CONFIGS
# ==============================

experiments = {
    "baseline": None,
    "balanced": "balanced",
}

results = []
best_score = -1.0

# ==============================
# TRAIN & VALIDATE
# ==============================

for name, class_weight in experiments.items():

    print(f"\n{'=' * 45}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 45}")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
                min_df=2,
                sublinear_tf=True,
            )
        ),
        (
            "svm",
            LinearSVC(
                C=1.0,
                class_weight=class_weight,
                random_state=SEED,
                max_iter=5000,
            )
        ),
    ])

    # Fit TF-IDF và SVM chỉ trên TRAIN
    pipeline.fit(X_train, y_train)

    # Dự đoán trên VALIDATION
    y_pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    macro_f1 = f1_score(
        y_val, y_pred, average="macro"
    )

    precision_1 = precision_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    )

    recall_1 = recall_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    )

    f1_1 = f1_score(
        y_val, y_pred, pos_label=1,
        zero_division=0
    )

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Reliable", "Unreliable"],
        digits=4,
        zero_division=0,
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ))

    results.append({
        "experiment": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
    })

    # Chọn mô hình dựa trên Macro F1
    if macro_f1 > best_score:
        best_score = macro_f1

        joblib.dump(
            pipeline,
            MODEL_DIR / "reintel_best_pipeline.joblib"
        )

        predictions = val_df.copy()
        predictions["predicted_label"] = y_pred

        predictions.to_csv(
            REPORT_DIR / "reintel_val_predictions.csv",
            index=False
        )

# ==============================
# SAVE EXPERIMENT RESULTS
# ==============================

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "reintel_baseline_results.csv",
    index=False
)

print("\n=== EXPERIMENT SUMMARY ===")
print(results_df.to_string(index=False))

print("\nBest Macro F1:", round(best_score, 4))
print("Training completed!")
