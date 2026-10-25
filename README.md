# Vietnamese Unreliable News Detection with TF-IDF and Linear SVM

Project nghiên cứu phân loại nội dung tiếng Việt theo độ tin cậy bằng **TF-IDF + Linear Support Vector Machine** trên hai bộ dữ liệu:

- **ReINTEL:** `0 = Reliable`, `1 = Unreliable`.
- **ViFN V1:** `0 = Real`, `1 = Fake`.

Project bao gồm quy trình audit dữ liệu, kiểm soát duplicate/near-duplicate, feature experiments, group-aware cross-validation, hyperparameter tuning, Final Test, error analysis, interpretability, learning curve và đánh giá cross-domain.

> Mô hình chỉ học tín hiệu thống kê từ văn bản và nhãn dataset. Đây không phải hệ thống fact-check và không tự xác minh thông tin bằng nguồn bên ngoài.

## Kết quả chính

| Model → Dataset | Loại đánh giá | Accuracy | Macro F1 | Precision lớp 1 | Recall lớp 1 | F1 lớp 1 |
|---|---|---:|---:|---:|---:|---:|
| ReINTEL → ReINTEL | In-domain | 0.8642 | 0.7708 | 0.5847 | 0.6699 | 0.6244 |
| ViFN → ViFN | In-domain | 0.8794 | 0.8788 | 0.9194 | 0.8261 | 0.8702 |
| ReINTEL → ViFN | Cross-domain | 0.6241 | 0.5875 | 0.7667 | 0.3333 | 0.4646 |
| ViFN → ReINTEL | Cross-domain | 0.7070 | 0.5944 | 0.2957 | 0.5340 | 0.3806 |

Kết quả cross-domain giảm mạnh so với in-domain, cho thấy model còn phụ thuộc vào nguồn dữ liệu, chủ đề, vocabulary và quy trình gán nhãn.

## Notebook

- [`01_reintel_experiments.ipynb`](notebooks/01_reintel_experiments.ipynb): toàn bộ thí nghiệm ReINTEL.
- [`02_vifn_experiments.ipynb`](notebooks/02_vifn_experiments.ipynb): toàn bộ thí nghiệm ViFN.
- [`03_comparative_analysis.ipynb`](notebooks/03_comparative_analysis.ipynb): so sánh in-domain và cross-domain.

Các notebook chủ yếu đọc artifact đã có, tính lại metric và dùng assertion để xác nhận dữ liệu, model và report đang đồng bộ.

## Cấu trúc project

```text
.
├── data/
│   ├── raw/                         # Dữ liệu nguồn, không commit
│   └── processed/                   # Dữ liệu đã xử lý, không commit
├── models/                          # Model joblib, không commit
├── notebooks/
│   ├── 01_reintel_experiments.ipynb
│   ├── 02_vifn_experiments.ipynb
│   └── 03_comparative_analysis.ipynb
├── reports/                         # Metrics, predictions và biểu đồ
├── scripts/
│   ├── 01_reintel_experiments/
│   ├── 02_vifn_experiments/
│   ├── 03_comparative_analysis/
│   └── README.md                    # Thứ tự chạy chi tiết
├── .gitignore
├── requirements.txt
└── README.md
```

## Yêu cầu

- Python 3.11 trở lên.
- Đủ RAM cho character TF-IDF và cosine similarity matrix.
- Kết nối Internet khi tải ReINTEL từ Hugging Face hoặc clone ViFN.

## Cài đặt

```bash
git clone https://github.com/mujin0422/svm-unreliable-news-detection.git
cd svm-unreliable-news-detection

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Windows PowerShell, kích hoạt môi trường bằng:

```powershell
.venv\Scripts\Activate.ps1
```

## Chuẩn bị dữ liệu

### ReINTEL

Các script inspect/prepare tự tải split có nhãn từ Hugging Face dataset `ReliableAI/ReINTEL`:

```bash
python scripts/01_reintel_experiments/01_inspect_reintel.py
python scripts/01_reintel_experiments/02_audit_reintel.py
python scripts/01_reintel_experiments/03_prepare_reintel.py
```

### ViFN V1

`data/raw/` được ignore. Clone ViFN V1 vào đúng vị trí:

```bash
git clone \
  https://github.com/huynhtuan0106/ViFN-Vietnamese_Fake_New_Datasets_Ver1.git \
  data/raw/vifn_v1
```

Sau đó kiểm tra và tạo Clean Train:

```bash
python scripts/02_vifn_experiments/01_inspect_vifn.py
python scripts/02_vifn_experiments/02_audit_vifn.py
python scripts/02_vifn_experiments/03_prepare_vifn.py
```

## Chạy workflow ReINTEL

Chạy lệnh từ thư mục gốc project:

```bash
python scripts/01_reintel_experiments/01_inspect_reintel.py
python scripts/01_reintel_experiments/02_audit_reintel.py
python scripts/01_reintel_experiments/03_prepare_reintel.py
python scripts/01_reintel_experiments/04_audit_near_duplicates.py
python scripts/01_reintel_experiments/05_review_near_duplicates.py
python scripts/01_reintel_experiments/06_prepare_reintel_v2.py
python scripts/01_reintel_experiments/07_review_mixed_groups.py
python scripts/01_reintel_experiments/08_prepare_reintel_v2_reviewed.py
python scripts/01_reintel_experiments/09_run_reintel_v2_experiments.py
python scripts/01_reintel_experiments/10_final_evaluate_reintel.py
python scripts/01_reintel_experiments/11_learning_curve_reintel.py
```

Các file `90–95_legacy_*` là thí nghiệm v1 được giữ để đối chiếu lịch sử, không phải nguồn của model ReINTEL cuối.

## Chạy workflow ViFN

```bash
python scripts/02_vifn_experiments/01_inspect_vifn.py
python scripts/02_vifn_experiments/02_audit_vifn.py
python scripts/02_vifn_experiments/03_prepare_vifn.py
python scripts/02_vifn_experiments/04_run_vifn_cv.py
python scripts/02_vifn_experiments/05_tune_vifn_svm.py
python scripts/02_vifn_experiments/06_final_evaluate_vifn.py
python scripts/02_vifn_experiments/07_analyze_vifn_errors.py
python scripts/02_vifn_experiments/08_explain_vifn_svm.py
python scripts/02_vifn_experiments/09_audit_vifn_source_bias.py
python scripts/02_vifn_experiments/10_learning_curve_vifn.py
```

## Chạy comparative analysis

Chỉ chạy sau khi model cuối của cả ReINTEL và ViFN đã tồn tại:

```bash
python scripts/03_comparative_analysis/01_evaluate_cross_dataset.py
```

## Mở notebook

```bash
jupyter lab
```

Mở notebook theo thứ tự `01` → `02` → `03`. Mỗi notebook tự tìm project root dựa trên các thư mục `data/`, `reports/` và `scripts/`.

## Bảo vệ Final Test

Hai script Final Test cố ý dừng nếu file kết quả đã tồn tại:

- `reports/reintel_v2_reviewed/final/test_metrics.json`
- `reports/vifn_v1/final/test_metrics.json`

Cơ chế này ngăn vô tình ghi đè benchmark hoặc lặp lại tuning dựa trên Test. Không xóa artifact chỉ để chạy lại và chọn kết quả tốt hơn. Khi thay đổi giao thức, nên tạo một version output mới.

## Artifact quan trọng

| Nội dung | Đường dẫn |
|---|---|
| ReINTEL feature experiments | `reports/reintel_v2_reviewed/fixed_experiments.csv` |
| ReINTEL tuning | `reports/reintel_v2_reviewed/svm_tuning_cv.csv` |
| ReINTEL Final Test | `reports/reintel_v2_reviewed/final/test_metrics.json` |
| ViFN feature experiments | `reports/vifn_v1/fixed_experiments_cv.csv` |
| ViFN tuning | `reports/vifn_v1/svm_tuning_cv.csv` |
| ViFN Final Test | `reports/vifn_v1/final/test_metrics.json` |
| Cross-domain comparison | `reports/cross_dataset/cross_dataset_results.csv` |

## Nguyên tắc đánh giá

- Chọn model bằng Macro F1, không chỉ Accuracy.
- TF-IDF được fit bên trong Pipeline để tránh leakage.
- Duplicate groups không được đi qua nhiều CV folds/splits.
- Final Test không dùng để chọn feature hoặc siêu tham số.
- Prediction CSV được giữ để có thể tính lại metric.
- Hệ số SVM chỉ giải thích quyết định của model, không chứng minh quan hệ nhân quả.

## Dataset references

- ReINTEL: Hugging Face `ReliableAI/ReINTEL`.
- ViFN V1: [ViFN Vietnamese Fake News Dataset Version 1](https://github.com/huynhtuan0106/ViFN-Vietnamese_Fake_New_Datasets_Ver1).
- ViFN paper: <https://doi.org/10.3837/tiis.2025.02.006>.
