
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed/reintel_v2")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load development data only
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")

# 2. Kiểm tra không có group trùng giữa train và val
assert set(train["group_id"]).isdisjoint(
    set(val["group_id"])
)

# 3. Gộp train và validation để audit
train["split"] = "train"
val["split"] = "val"

df = pd.concat(
    [train, val],
    ignore_index=True
)

# 4. Tìm group có nhiều nhãn
label_counts = df.groupby(
    "group_id"
)["label"].nunique()

mixed_ids = label_counts[
    label_counts > 1
].index

print("=== MIXED-LABEL AUDIT ===")
print("Development samples:", len(df))
print("Mixed-label groups:", len(mixed_ids))

# 5. Lấy tất cả mẫu thuộc group mâu thuẫn
mixed_df = df[
    df["group_id"].isin(mixed_ids)
].copy()

# 6. In thống kê từng group
for group_id, group in mixed_df.groupby("group_id"):

    print("\n" + "=" * 60)
    print("Group ID:", group_id)
    print("Samples:", len(group))
    print("Label distribution:")
    print(group["label"].value_counts())

    for _, row in group.iterrows():
        print("\nSplit:", row["split"])
        print("Label:", row["label"])
        print("Text length:", len(str(row["text"])))
        print("Text preview:")
        print(str(row["text"])[:700])
        print("-" * 40)

# 7. Save full text for manual review
mixed_df.to_csv(
    REPORT_DIR / "reintel_v2_mixed_groups_review.csv",
    index=False
)

print("\nAudit completed!")
