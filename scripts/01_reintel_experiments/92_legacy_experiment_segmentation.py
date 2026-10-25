
from pathlib import Path
import pandas as pd

from underthesea import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

X_train = train["text"].fillna("").tolist()
X_val = val["text"].fillna("").tolist()

y_train = train["label"]
y_val = val["label"]

# 2. Vietnamese word segmentation
def segment_text(text):
    return word_tokenize(
        text,
        format="text",
        use_token_normalize=False,
    )

print("Segmenting training data...")
X_train_segmented = [
    segment_text(text) for text in X_train
]

print("Segmenting validation data...")
X_val_segmented = [
    segment_text(text) for text in X_val
]

# 3. Experiments
experiments = {
    "without_segmentation": (
        X_train,
        X_val,
    ),
    "with_segmentation": (
        X_train_segmented,
        X_val_segmented,
    ),
}

results = []

for name, (train_texts, val_texts) in experiments.items():

    print(f"\n=== {name} ===")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
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

    model.fit(train_texts, y_train)
    predictions = model.predict(val_texts)

    result = {
        "experiment": name,
        "accuracy": accuracy_score(
            y_val, predictions
        ),
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
        "n_features": len(
            model.named_steps["tfidf"]
            .get_feature_names_out()
        ),
    }

    results.append(result)

    print(result)
    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ))

# 4. Save results
df = pd.DataFrame(results)
df.to_csv(
    REPORT_DIR / "experiment_segmentation.csv",
    index=False
)

print("\n=== FINAL COMPARISON ===")
print(df.to_string(index=False))
