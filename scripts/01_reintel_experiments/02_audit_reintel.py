import pandas as pd
from datasets import load_dataset

dataset = load_dataset("ReliableAI/ReINTEL", split="train")

df = dataset.select_columns(
    ["post_message", "label"]
).to_pandas()

df["text"] = (
    df["post_message"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 1. Kiểm tra dữ liệu rỗng
print("=== EMPTY TEXT ===")
print("Số văn bản rỗng:", (df["text"] == "").sum())

# 2. Kiểm tra bài chỉ chứa URL
url_only = df["text"].str.fullmatch(
    r"(?:<URL>\s*)+",
    case=False
).fillna(False)

print("\n=== URL ONLY ===")
print("Số bài chỉ chứa <URL>:", url_only.sum())

# 3. Kiểm tra trùng lặp
print("\n=== DUPLICATES ===")
print("Số bản ghi trùng text:", df["text"].duplicated().sum())

# 4. Kiểm tra văn bản trùng nhưng khác nhãn
label_counts = df.groupby("text")["label"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print("\n=== LABEL CONFLICTS ===")
print("Số văn bản bị mâu thuẫn nhãn:", len(conflicting_texts))

# 5. Thống kê độ dài theo nhãn
df["length"] = df["text"].str.len()

print("\n=== LENGTH BY LABEL ===")
print(df.groupby("label")["length"].describe())

# 6. Kiểm tra bài quá ngắn / quá dài
print("\n=== LENGTH OUTLIERS ===")
print("Số bài < 20 ký tự:", (df["length"] < 20).sum())
print("Số bài > 5000 ký tự:", (df["length"] > 5000).sum())
