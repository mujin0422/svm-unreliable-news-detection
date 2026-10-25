
from pathlib import Path
import json
import hashlib
import unicodedata

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==================================
# CONFIG
# ==================================

DATA_DIR = Path("data/processed/vifn_v1")
RAW_DIR = Path("data/raw/vifn_v1")
MODEL_DIR = Path("models/vifn_v1")
REPORT_DIR = Path("reports/vifn_v1/final")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "tuned_word_bigram.joblib"
RESULT_PATH = REPORT_DIR / "test_metrics.json"

# Không vô tình ghi đè kết quả Test
if RESULT_PATH.exists():
    raise SystemExit(
        "Final Test đã được đánh giá. "
        "Không ghi đè kết quả cũ."
    )

# ==================================
# 1. LOAD DATA
# ==================================

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(RAW_DIR / "test_data.csv")

assert len(train) == 924
assert len(test) == 282

assert test["content"].notna().all()
assert test["label"].isin([0, 1]).all()

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()

test["text"] = test["content"].map(normalize_text)

# Kiểm tra không trùng Train
assert set(train["text"]).isdisjoint(
    set(test["text"])
)

X_test = test["text"]
y_test = test["label"]

print("=== DATASET ===")
print("Train samples:", len(train))
print("Test samples:", len(test))

print("\nTest label distribution:")
print(y_test.value_counts())

# ==================================
# 2. LOAD TRAINED MODEL
# ==================================

# Chỉ load model do chính mình tạo ra
model = joblib.load(MODEL_PATH)

assert model.named_steps["svm"].C == 1.0
assert model.named_steps["svm"].class_weight == "balanced"
assert model.named_steps["tfidf"].ngram_range == (1, 2)

print("\nLoaded model:", MODEL_PATH)

# ==================================
# 3. FINAL TEST
# ==================================

predictions = model.predict(X_test)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "precision_fake": float(
            precision_score(
                y_true, y_pred,
                pos_label=1, zero_division=0
            )
        ),
        "recall_fake": float(
            recall_score(
                y_true, y_pred,
                pos_label=1, zero_division=0
            )
        ),
        "f1_fake": float(
            f1_score(
                y_true, y_pred,
                pos_label=1, zero_division=0
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

metrics = calculate_metrics(y_test, predictions)

print("\n=== FINAL TEST RESULTS ===")

print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    digits=4,
    zero_division=0,
))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(
    y_test, predictions, labels=[0, 1]
))

print("\n=== FINAL METRICS ===")
print(json.dumps(metrics, indent=2))

# ==================================
# 4. SECONDARY DUPLICATE ANALYSIS
# ==================================

# Chỉ phân tích, không thay đổi Test gốc
test_results = test.copy()
test_results["predicted_label"] = predictions

unique_test = test_results.drop_duplicates(
    subset=["text"],
    keep="first",
)

conflicting_duplicates = (
    test_results.groupby("text")["label"]
    .nunique()
    .gt(1)
    .sum()
)

unique_metrics = calculate_metrics(
    unique_test["label"],
    unique_test["predicted_label"],
)

print("\n=== UNIQUE TEST ANALYSIS ===")
print("Unique samples:", len(unique_test))
print("Duplicate label conflicts:", conflicting_duplicates)
print("Macro F1:", round(unique_metrics["macro_f1"], 4))
print("F1 Fake:", round(unique_metrics["f1_fake"], 4))

# ==================================
# 5. SAVE RESULTS
# ==================================

output = {
    "dataset": "ViFN V1",
    "protocol": "Clean Train + Original Test",
    "train_samples": len(train),
    "test_samples": len(test),
    "model": "TF-IDF + LinearSVC",
    "C": 1.0,
    "class_weight": "balanced",
    "ngram_range": [1, 2],
    "cv_macro_f1": 0.835794,
    "test_sha256": hashlib.sha256(
        (RAW_DIR / "test_data.csv").read_bytes()
    ).hexdigest(),
    "official_test_metrics": metrics,
    "unique_test_samples": len(unique_test),
    "unique_test_metrics": unique_metrics,
    "duplicate_label_conflicts": int(
        conflicting_duplicates
    ),
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        output, f,
        ensure_ascii=False,
        indent=2,
    )

pd.DataFrame({
    "sample_index": range(len(test)),
    "actual_label": y_test,
    "predicted_label": predictions,
}).to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

print("\nFinal evaluation completed!")
print("Results saved:", RESULT_PATH)
