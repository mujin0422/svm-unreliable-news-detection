
from pathlib import Path
from shutil import copy2
import hashlib
import pandas as pd

INPUT_DIR = Path("data/processed/reintel_v2")
OUTPUT_DIR = Path("data/processed/reintel_v2_reviewed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Các nhóm được chọn để loại khỏi dữ liệu phát triển
EXCLUDED_GROUPS = {803, 946, 3758}

# 1. Load dữ liệu
train = pd.read_csv(INPUT_DIR / "train.csv")
val = pd.read_csv(INPUT_DIR / "val.csv")
test = pd.read_csv(INPUT_DIR / "test.csv")

# 2. Kiểm tra nhãn mâu thuẫn trong development
dev = pd.concat([
    train.assign(split="train"),
    val.assign(split="val"),
], ignore_index=True)

mixed_counts = dev.groupby("group_id")["label"].nunique()
mixed_groups = set(
    mixed_counts[mixed_counts > 1].index
)

assert mixed_groups == EXCLUDED_GROUPS, (
    f"Unexpected mixed groups: {mixed_groups}"
)

# Test phải độc lập với các nhóm này
assert set(test["group_id"]).isdisjoint(EXCLUDED_GROUPS)

# 3. Lưu metadata về những mẫu bị loại
removed = dev[
    dev["group_id"].isin(EXCLUDED_GROUPS)
].copy()

removed["text_length"] = removed["text"].fillna("").str.len()

removed[
    ["split", "group_id", "label", "text_length"]
].to_csv(
    OUTPUT_DIR / "excluded_samples_audit.csv",
    index=False
)

# 4. Loại nhóm mâu thuẫn khỏi Train và Val
train_clean = train[
    ~train["group_id"].isin(EXCLUDED_GROUPS)
].copy()

val_clean = val[
    ~val["group_id"].isin(EXCLUDED_GROUPS)
].copy()

# 5. Kiểm tra dữ liệu
assert len(train) - len(train_clean) == 4
assert len(val) - len(val_clean) == 3

assert set(train_clean["group_id"]).isdisjoint(
    set(val_clean["group_id"])
)
assert set(train_clean["group_id"]).isdisjoint(
    set(test["group_id"])
)
assert set(val_clean["group_id"]).isdisjoint(
    set(test["group_id"])
)

# 6. Lưu dữ liệu phát triển mới
train_clean.to_csv(
    OUTPUT_DIR / "train.csv", index=False
)

val_clean.to_csv(
    OUTPUT_DIR / "val.csv", index=False
)

# 7. Sao chép Test nguyên vẹn, không xử lý lại
source_test = INPUT_DIR / "test.csv"
target_test = OUTPUT_DIR / "test.csv"

copy2(source_test, target_test)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha256(source_test) == sha256(target_test)

# 8. In kết quả
for name, frame in [
    ("TRAIN", train_clean),
    ("VAL", val_clean),
    ("TEST", test),
]:
    print(f"\n=== {name} ===")
    print("Samples:", len(frame))
    print("Label distribution:")
    print(frame["label"].value_counts())
    print("Groups:", frame["group_id"].nunique())

print("\nExcluded samples:", len(removed))
print("Excluded groups:", sorted(EXCLUDED_GROUPS))
print("Test integrity verified: SHA256 identical")
print("\nReINTEL v2 Reviewed completed!")
