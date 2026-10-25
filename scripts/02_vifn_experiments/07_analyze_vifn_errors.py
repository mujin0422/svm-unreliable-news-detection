
from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/vifn_v1")
MODEL_DIR = Path("models/vifn_v1")
REPORT_DIR = Path("reports/vifn_v1/final")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()


# =====================================
# 1. LOAD FINAL TEST RESULTS
# =====================================

test = pd.read_csv(
    RAW_DIR / "test_data.csv"
)

predictions = pd.read_csv(
    REPORT_DIR / "test_predictions.csv"
)

assert len(test) == len(predictions) == 282
assert predictions["sample_index"].tolist() == list(
    range(len(test))
)

# Verify saved labels match the original Test
assert np.array_equal(
    test["label"].to_numpy(),
    predictions["actual_label"].to_numpy()
)

df = pd.DataFrame({
    "text": test["content"].map(normalize_text),
    "actual": predictions["actual_label"],
    "predicted": predictions["predicted_label"],
})

# =====================================
# 2. LOAD FROZEN MODEL
# =====================================

model = joblib.load(
    MODEL_DIR / "tuned_word_bigram.joblib"
)

# Positive score favors class 1 (Fake)
df["decision_score"] = model.decision_function(
    df["text"]
)

assert np.array_equal(
    model.predict(df["text"]),
    df["predicted"].to_numpy()
)

df["length"] = df["text"].str.len()

# =====================================
# 3. CLASSIFY ERRORS
# =====================================

conditions = [
    (df["actual"] == 0) & (df["predicted"] == 0),
    (df["actual"] == 0) & (df["predicted"] == 1),
    (df["actual"] == 1) & (df["predicted"] == 0),
    (df["actual"] == 1) & (df["predicted"] == 1),
]

choices = ["TN", "FP", "FN", "TP"]

df["error_type"] = np.select(
    conditions,
    choices,
    default="UNKNOWN"
)

errors = df[
    df["error_type"].isin(["FP", "FN"])
].copy()

# Scores further from 0 correspond to
# stronger SVM classification margins
errors["margin_magnitude"] = (
    errors["decision_score"].abs()
)

# =====================================
# 4. ERROR STATISTICS
# =====================================

print("=== ERROR DISTRIBUTION ===")
print(df["error_type"].value_counts())

print("\n=== TEXT LENGTH BY ERROR TYPE ===")
print(
    df.groupby("error_type")["length"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\n=== ERROR RATE BY TEXT LENGTH ===")

df["length_group"] = pd.cut(
    df["length"],
    bins=[0, 80, 150, 350, float("inf")],
    labels=[
        "1-80",
        "81-150",
        "151-350",
        ">350"
    ],
)

df["is_error"] = df["actual"] != df["predicted"]

length_stats = df.groupby(
    "length_group", observed=True
).agg(
    total=("is_error", "size"),
    errors=("is_error", "sum"),
    error_rate=("is_error", "mean"),
)

print(length_stats)

# =====================================
# 5. MOST CONFIDENT MISTAKES
# =====================================

errors = errors.sort_values(
    "margin_magnitude",
    ascending=False
)

print("\n=== TOP FALSE NEGATIVES ===")

fn = errors[errors["error_type"] == "FN"]

for _, row in fn.head(5).iterrows():
    print("\nScore:", round(row["decision_score"], 4))
    print("Text:", row["text"][:500])
    print("-" * 50)

print("\n=== TOP FALSE POSITIVES ===")

fp = errors[errors["error_type"] == "FP"]

for _, row in fp.head(5).iterrows():
    print("\nScore:", round(row["decision_score"], 4))
    print("Text:", row["text"][:500])
    print("-" * 50)

# =====================================
# 6. SAVE RESULTS
# =====================================

errors.to_csv(
    REPORT_DIR / "error_analysis.csv",
    index=False
)

length_stats.to_csv(
    REPORT_DIR / "error_by_length.csv"
)

print("\n=== SUMMARY ===")
print("Total test:", len(df))
print("Total errors:", len(errors))
print("False positives:", len(fp))
print("False negatives:", len(fn))

print("\nError analysis completed!")
