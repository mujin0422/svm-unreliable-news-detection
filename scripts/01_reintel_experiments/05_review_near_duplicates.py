
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed/reintel")
REPORT_DIR = Path("reports")

THRESHOLD = 0.95

# Load các tập dữ liệu
datasets = {
    "train": pd.read_csv(DATA_DIR / "train.csv"),
    "val": pd.read_csv(DATA_DIR / "val.csv"),
    "test": pd.read_csv(DATA_DIR / "test.csv"),
}

audit = pd.read_csv(
    REPORT_DIR / "near_duplicate_audit.csv"
)

# Chỉ giữ các cặp tương đồng rất cao
suspects = audit[
    audit["similarity"] >= THRESHOLD
].sort_values("similarity", ascending=False)

records = []

for _, pair in suspects.iterrows():
    ref_split = pair["reference_split"]
    target_split = pair["target_split"]

    ref_idx = int(pair["nearest_reference_index"])
    target_idx = int(pair["target_index"])

    ref = datasets[ref_split].iloc[ref_idx]
    target = datasets[target_split].iloc[target_idx]

    # Không sử dụng nhãn test trong bước audit
    ref_label = (
        ref["label"] if ref_split != "test"
        else None
    )

    target_label = (
        target["label"] if target_split != "test"
        else None
    )

    records.append({
        "reference_split": ref_split,
        "target_split": target_split,
        "similarity": pair["similarity"],
        "reference_index": ref_idx,
        "target_index": target_idx,
        "reference_label": ref_label,
        "target_label": target_label,
        "reference_text": ref["text"],
        "target_text": target["text"],
    })

review = pd.DataFrame(records)

review.to_csv(
    REPORT_DIR / "near_duplicate_pairs.csv",
    index=False,
)

# In tối đa 5 cặp có similarity cao nhất
# cho mỗi tổ hợp tập dữ liệu
for (ref_split, target_split), group in review.groupby(
    ["reference_split", "target_split"]
):
    print(f"\n=== {target_split} vs {ref_split} ===")
    print("Total suspicious pairs:", len(group))

    for _, row in group.head(5).iterrows():
        print("\nSimilarity:", round(row["similarity"], 5))

        print("\nREFERENCE:")
        print(str(row["reference_text"])[:500])

        print("\nTARGET:")
        print(str(row["target_text"])[:500])

        if target_split != "test":
            print(
                "\nLabels:",
                row["reference_label"],
                row["target_label"],
            )

        print("-" * 60)

print("\nReview file saved.")
