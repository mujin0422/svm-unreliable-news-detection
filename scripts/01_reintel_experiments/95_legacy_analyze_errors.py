
from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports")

df = pd.read_csv(
    REPORT_DIR / "reintel_val_predictions.csv"
)

# Chỉ lấy dự đoán sai
errors = df[
    df["label"] != df["predicted_label"]
].copy()

# Phân loại lỗi
errors["error_type"] = errors.apply(
    lambda row: "False Positive"
    if row["label"] == 0
    else "False Negative",
    axis=1
)

# Thống kê độ dài
errors["text_length"] = (
    errors["text"].fillna("").str.len()
)

print("=== ERROR DISTRIBUTION ===")
print(errors["error_type"].value_counts())

print("\n=== TEXT LENGTH BY ERROR TYPE ===")
print(
    errors.groupby("error_type")["text_length"]
    .describe()
)

# Hiển thị các trường hợp dự đoán sai
for error_type in ["False Positive", "False Negative"]:

    subset = errors[
        errors["error_type"] == error_type
    ]

    print(f"\n=== {error_type.upper()} ===")

    for _, row in subset.head(5).iterrows():
        print("\nActual:", row["label"])
        print("Predicted:", row["predicted_label"])
        print("Length:", row["text_length"])
        print("Text:", row["text"][:500])
        print("-" * 50)

# Lưu toàn bộ kết quả để phân tích thủ công
errors.to_csv(
    REPORT_DIR / "reintel_validation_errors.csv",
    index=False
)

print("\nError analysis completed!")
