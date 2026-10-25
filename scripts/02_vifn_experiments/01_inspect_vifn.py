
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/vifn_v1")

# Chỉ khảo sát tập train
df = pd.read_csv(DATA_DIR / "train_data.csv")

print("=== VIFN INITIAL EDA ===")
print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

print("\n=== LABEL DISTRIBUTION ===")
print(df["label"].value_counts(dropna=False))
print(df["label"].value_counts(normalize=True))

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Chuẩn hóa khoảng trắng để kiểm tra
df["text"] = (
    df["content"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\n=== DATA QUALITY ===")
print("Empty text:", (df["text"] == "").sum())
print("Duplicate text:", df["text"].duplicated().sum())

# Kiểm tra nhóm văn bản có nhãn khác nhau
conflicts = (
    df.groupby("text")["label"]
    .nunique()
)

print(
    "Conflicting label groups:",
    (conflicts > 1).sum()
)

# Độ dài văn bản
df["length"] = df["text"].str.len()

print("\n=== TEXT LENGTH ===")
print(df["length"].describe())

print("\n=== LENGTH BY LABEL ===")
print(df.groupby("label")["length"].describe())

print("\n=== SAMPLE TEXTS ===")
for _, row in df.head(3).iterrows():
    print("\nLabel:", row["label"])
    print("Text:", row["text"][:300])
    print("-" * 50)
