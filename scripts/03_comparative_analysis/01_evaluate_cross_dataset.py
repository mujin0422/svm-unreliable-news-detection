
from pathlib import Path
import unicodedata
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/cross_dataset")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()


# Load final models
models = {
    "ReINTEL": joblib.load(
        MODEL_DIR / "reintel_v2_reviewed/reintel_final.joblib"
    ),
    "ViFN": joblib.load(
        MODEL_DIR / "vifn_v1/tuned_word_bigram.joblib"
    ),
}

# Load held-out test sets
reintel = pd.read_csv(
    "data/processed/reintel_v2_reviewed/test.csv"
)

vifn = pd.read_csv(
    "data/raw/vifn_v1/test_data.csv"
)

vifn["text"] = vifn["content"].map(normalize_text)

datasets = {
    "ReINTEL": reintel,
    "ViFN": vifn,
}

results = []

for source_name, model in models.items():
    for target_name, df in datasets.items():

        print(f"\n=== {source_name} -> {target_name} ===")

        X = df["text"].fillna("")
        y = df["label"]

        predictions = model.predict(X)

        cm = confusion_matrix(
            y, predictions, labels=[0, 1]
        )

        print("Confusion Matrix:")
        print(cm)

        result = {
            "source_model": source_name,
            "target_dataset": target_name,
            "evaluation_type": (
                "in_domain"
                if source_name == target_name
                else "cross_domain"
            ),
            "accuracy": accuracy_score(y, predictions),
            "macro_f1": f1_score(
                y, predictions, average="macro"
            ),
            "precision_1": precision_score(
                y, predictions, pos_label=1,
                zero_division=0
            ),
            "recall_1": recall_score(
                y, predictions, pos_label=1,
                zero_division=0
            ),
            "f1_1": f1_score(
                y, predictions, pos_label=1,
                zero_division=0
            ),
        }

        results.append(result)
        print(result)

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "cross_dataset_results.csv",
    index=False
)

print("\n=== FINAL CROSS-DATASET COMPARISON ===")
print(results_df.to_string(index=False))
