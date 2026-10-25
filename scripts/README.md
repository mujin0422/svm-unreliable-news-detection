# Quy trình chạy scripts

Thư mục `scripts/` được chia theo ba notebook chính. Tất cả lệnh nên được chạy từ thư mục gốc của project để các đường dẫn tương đối tới `data/`, `reports/` và `models/` hoạt động đúng.

## 1. ReINTEL — `01_reintel_experiments/`

Luồng ReINTEL v2 Reviewed hiện tại:

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

Các file `90–95_legacy_*` là thí nghiệm v1 được giữ để đối chiếu lịch sử. Chúng không phải nguồn của model ReINTEL cuối.

Lưu ý: `10_final_evaluate_reintel.py` cố ý dừng nếu `test_metrics.json` đã tồn tại để bảo vệ kết quả Test khỏi bị ghi đè.

## 2. VIFN — `02_vifn_experiments/`

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

Các bước 07–10 là phân tích hậu nghiệm/bổ sung; không dùng Final Test để chọn lại cấu hình.

## 3. Comparative Analysis — `03_comparative_analysis/`

Chỉ chạy sau khi model cuối của ReINTEL và VIFN đã tồn tại:

```bash
python scripts/03_comparative_analysis/01_evaluate_cross_dataset.py
```

## Quy ước đặt tên

- `01`, `02`, ...: thứ tự chính trong workflow.
- `90`, `91`, ... + `legacy`: thí nghiệm cũ, chỉ dùng để đối chiếu.
- Tên folder khớp với notebook tương ứng trong `notebooks/`.
