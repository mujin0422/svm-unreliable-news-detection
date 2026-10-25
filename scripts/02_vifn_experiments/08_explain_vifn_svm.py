
from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/vifn_v1")
MODEL_PATH = Path(
    "models/vifn_v1/tuned_word_bigram.joblib"
)
PRED_PATH = Path(
    "reports/vifn_v1/final/test_predictions.csv"
)
OUTPUT_DIR = Path("reports/vifn_v1/interpretability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split()).strip()


# =====================================
# 1. LOAD FROZEN MODEL
# =====================================

model = joblib.load(MODEL_PATH)

vectorizer = model.named_steps["tfidf"]
svm = model.named_steps["svm"]

assert list(svm.classes_) == [0, 1]

features = vectorizer.get_feature_names_out()
weights = svm.coef_[0]

print("=== MODEL ===")
print("Total features:", len(features))
print("Intercept:", svm.intercept_[0])


# =====================================
# 2. GLOBAL FEATURE IMPORTANCE
# =====================================

TOP_N = 30

fake_indices = np.argsort(weights)[-TOP_N:][::-1]
real_indices = np.argsort(weights)[:TOP_N]

global_records = []

for label, indices in [
    ("Fake", fake_indices),
    ("Real", real_indices),
]:
    for rank, idx in enumerate(indices, start=1):
        global_records.append({
            "class_direction": label,
            "rank": rank,
            "feature": features[idx],
            "coefficient": float(weights[idx]),
        })

global_df = pd.DataFrame(global_records)

global_df.to_csv(
    OUTPUT_DIR / "global_features.csv",
    index=False,
)

print("\n=== TOP FEATURES SUPPORTING FAKE ===")
print(
    global_df[
        global_df["class_direction"] == "Fake"
    ].head(15).to_string(index=False)
)

print("\n=== TOP FEATURES SUPPORTING REAL ===")
print(
    global_df[
        global_df["class_direction"] == "Real"
    ].head(15).to_string(index=False)
)


# =====================================
# 3. LOAD EXISTING TEST PREDICTIONS
# =====================================

test = pd.read_csv(
    RAW_DIR / "test_data.csv"
)

predictions = pd.read_csv(PRED_PATH)

assert len(test) == len(predictions) == 282
assert predictions["sample_index"].tolist() == list(
    range(len(test))
)
assert np.array_equal(
    test["label"].to_numpy(),
    predictions["actual_label"].to_numpy()
)

texts = test["content"].map(normalize_text)

X = vectorizer.transform(texts)

# Confirm that the frozen model produces
# the same predictions as the final test
current_predictions = model.predict(texts)

assert np.array_equal(
    current_predictions,
    predictions["predicted_label"].to_numpy()
)

scores = model.decision_function(texts)


# =====================================
# 4. LOCAL FEATURE CONTRIBUTIONS
# =====================================

def explain_sample(index, top_n=8):
    row = X.getrow(index)

    indices = row.indices
    contributions = row.data * weights[indices]

    positive = np.where(contributions > 0)[0]
    negative = np.where(contributions < 0)[0]

    positive = positive[
        np.argsort(contributions[positive])[::-1]
    ][:top_n]

    negative = negative[
        np.argsort(contributions[negative])
    ][:top_n]

    def describe(selected):
        return " | ".join(
            f"{features[indices[k]]}: "
            f"{contributions[k]:+.4f}"
            for k in selected
        )

    return describe(positive), describe(negative)


records = []

for index in range(len(test)):
    actual = int(predictions.iloc[index]["actual_label"])
    predicted = int(
        predictions.iloc[index]["predicted_label"]
    )

    if actual == predicted:
        continue

    supporting_fake, supporting_real = (
        explain_sample(index)
    )

    error_type = (
        "FN" if actual == 1 else "FP"
    )

    records.append({
        "sample_index": index,
        "error_type": error_type,
        "actual": actual,
        "predicted": predicted,
        "decision_score": float(scores[index]),
        "text": texts.iloc[index],
        "supporting_fake": supporting_fake,
        "supporting_real": supporting_real,
    })


local_df = pd.DataFrame(records)

local_df["margin_magnitude"] = (
    local_df["decision_score"].abs()
)

local_df = local_df.sort_values(
    "margin_magnitude",
    ascending=False,
)

local_df.to_csv(
    OUTPUT_DIR / "error_explanations.csv",
    index=False,
)


# =====================================
# 5. PRINT MOST IMPORTANT ERRORS
# =====================================

for error_type in ["FN", "FP"]:
    subset = local_df[
        local_df["error_type"] == error_type
    ].head(3)

    print(f"\n=== TOP {error_type} EXPLANATIONS ===")

    for _, row in subset.iterrows():
        print("\nSample index:", row["sample_index"])
        print("Decision score:", row["decision_score"])
        print("Text:", row["text"][:350])

        print(
            "\nFeatures supporting Fake:",
            row["supporting_fake"]
        )

        print(
            "\nFeatures supporting Real:",
            row["supporting_real"]
        )

        print("-" * 60)


print("\n=== SUMMARY ===")
print("Total explained errors:", len(local_df))
print("Global features saved:", len(global_df))

print("\nModel interpretability completed!")
