
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
OUTPUT_DIR = Path("data/processed/reintel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dữ liệu có nhãn
dataset = load_dataset("ReliableAI/ReINTEL", split="train")

df = dataset.select_columns(
    ["post_message", "label"]
).to_pandas()

df = df.rename(columns={"post_message": "text"})

print("Original:", len(df))

# 2. Chuẩn hóa khoảng trắng
df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 3. Loại bỏ văn bản rỗng
df = df[df["text"] != ""].copy()
print("After removing empty:", len(df))

# 4. Loại bỏ bài chỉ chứa URL placeholder
url_only = df["text"].str.fullmatch(
    r"(?:<URL>\s*)+",
    case=False,
    na=False
)

df = df[~url_only].copy()
print("After removing URL only:", len(df))

# 5. Kiểm tra các nhóm mâu thuẫn nhãn
label_counts = df.groupby("text")["label"].nunique()

conflicting_texts = label_counts[
    label_counts > 1
].index

print("Conflicting text groups:", len(conflicting_texts))

# In các nhóm mâu thuẫn để kiểm tra
conflicts = df[df["text"].isin(conflicting_texts)]

for text, group in conflicts.groupby("text"):
    print("\nConflicting text:", repr(text[:200]))
    print("Labels:", group["label"].tolist())

# Loại tất cả mẫu thuộc nhóm mâu thuẫn nhãn
df = df[~df["text"].isin(conflicting_texts)].copy()

# 6. Loại bản ghi trùng văn bản
df = df.drop_duplicates(subset=["text"])

# 7. Chỉ giữ cột cần thiết
df = df[["text", "label"]].reset_index(drop=True)

print("\nClean dataset:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())

# 8. Chia train 70%, temp 30%
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

# 9. Chia temp thành validation và test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

# 10. Kiểm tra không rò rỉ văn bản trùng lặp
train_texts = set(train_df["text"])
val_texts = set(val_df["text"])
test_texts = set(test_df["text"])

assert train_texts.isdisjoint(val_texts)
assert train_texts.isdisjoint(test_texts)
assert val_texts.isdisjoint(test_texts)

# 11. Lưu dữ liệu
for name, split in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    split.to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False
    )

    print(f"\n{name.upper()}: {len(split)}")
    print(split["label"].value_counts(normalize=True))

print("\nData preparation completed!")
