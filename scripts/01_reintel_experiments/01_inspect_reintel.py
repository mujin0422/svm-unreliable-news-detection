import pandas as pd
from datasets import load_dataset

# 1. Load dữ liệu có nhãn
dataset = load_dataset("ReliableAI/ReINTEL", split="train")

# 2. Chỉ lấy văn bản và nhãn
df = dataset.select_columns(
    ["post_message", "label"]
).to_pandas()

# 3. Thông tin cơ bản
print("Số lượng mẫu:", len(df))
print("\nCác cột:", df.columns.tolist())

print("\nPhân bố nhãn:")
print(df["label"].value_counts())

print("\nGiá trị bị thiếu:")
print(df.isnull().sum())

# 4. Kiểm tra trùng lặp
print("\nSố văn bản trùng lặp:")
print(df["post_message"].duplicated().sum())

# 5. Độ dài văn bản
df["text_length"] = (
    df["post_message"].fillna("").str.len()
)

print("\nThống kê độ dài văn bản:")
print(df["text_length"].describe())

# 6. Xem một số mẫu
print("\nDữ liệu mẫu:")
print(df[["post_message", "label"]].head())
