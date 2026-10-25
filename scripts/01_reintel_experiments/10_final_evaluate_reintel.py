
from pathlib import Path
import json
import hashlib

import joblib
import pandas as pd

from sklearn import __version__ as sklearn_version
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

SEED = 42

DATA_DIR = Path("data/processed/reintel_v2_reviewed")
REPORT_DIR = Path("reports/reintel_v2_reviewed/final")
MODEL_DIR = Path("models/reintel_v2_reviewed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORT_DIR / "test_metrics.json"

# Bảo vệ kết quả test khỏi việc vô tình ghi đè
if METRICS_PATH.exists():
    raise SystemExit(
        "Final test already evaluated. "
        "Existing results will not be overwritten."
    )

# ==================================
# 1. LOAD DATA
# ==================================

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

dev = pd.concat([train, val], ignore_index=True)

# ==================================
# 2. DATA INTEGRITY CHECKS
# ==================================

assert len(dev) == 3663
assert len(test) == 611

assert set(dev["group_id"]).isdisjoint(
    set(test["group_id"])
)

assert dev.groupby("group_id")["label"].nunique().max() == 1

assert set(dev["label"].unique()) <= {0, 1}
assert set(test["label"].unique()) <= {0, 1}

X_dev = dev["text"].fillna("")
y_dev = dev["label"]

X_test = test["text"].fillna("")
y_test = test["label"]

print("Development samples:", len(dev))
print("Test samples:", len(test))

# ==================================
# 3. FINAL MODEL
# ==================================

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

# Refit trên Train + Validation
print("\nTraining final model...")
model.fit(X_dev, y_dev)

# ==================================
# 4. FINAL TEST EVALUATION
# ==================================

print("\nEvaluating test set...")

y_pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "macro_f1": float(f1_score(
        y_test, y_pred, average="macro"
    )),
    "precision_1": float(precision_score(
        y_test, y_pred, pos_label=1,
        zero_division=0
    )),
    "recall_1": float(recall_score(
        y_test, y_pred, pos_label=1,
        zero_division=0
    )),
    "f1_1": float(f1_score(
        y_test, y_pred, pos_label=1,
        zero_division=0
    )),
}

cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

metrics.update({
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
})

print("\n=== FINAL TEST RESULTS ===")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Reliable", "Unreliable"],
    digits=4,
    zero_division=0,
))

print("\nConfusion Matrix:")
print(cm)

print("\n=== FINAL METRICS ===")
print(json.dumps(metrics, indent=2))

# ==================================
# 5. SAVE RESULTS & MODEL
# ==================================

model_path = MODEL_DIR / "reintel_final.joblib"
joblib.dump(model, model_path)

test_predictions = pd.DataFrame({
    "sample_index": range(len(test)),
    "actual_label": y_test.to_numpy(),
    "predicted_label": y_pred,
})

test_predictions.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

metadata = {
    "dataset": "ReINTEL v2 Reviewed",
    "train_samples": len(dev),
    "test_samples": len(test),
    "model": "TF-IDF + LinearSVC",
    "C": 0.1,
    "class_weight": "balanced",
    "ngram_range": [1, 2],
    "selection_metric": "CV Macro F1",
    "sklearn_version": sklearn_version,
    "test_sha256": hashlib.sha256(
        (DATA_DIR / "test.csv").read_bytes()
    ).hexdigest(),
    "metrics": metrics,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        metadata, f,
        ensure_ascii=False,
        indent=2
    )

print("\nFinal model saved:", model_path)
print("Results saved:", METRICS_PATH)
print("\nFinal evaluation completed!")
