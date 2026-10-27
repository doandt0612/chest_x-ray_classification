# Thẻ mô hình: phân loại X-quang ngực v1.0.0

Bản đầu, 26/10/2026. Số liệu lấy từ `reports/evaluation/v1.0.0/`.

## Mục đích

- Phân loại ảnh X-quang ngực thẳng của trẻ em thành NORMAL hoặc PNEUMONIA (viêm phổi).
- Chỉ dùng cho học tập và nghiên cứu, trong đồ án MLOps. **Không dùng để chẩn đoán.**
- Đầu ra là xác suất PNEUMONIA và một nhãn theo ngưỡng cố định.

## Mô hình

| Mục | Giá trị |
| :-- | :-- |
| Phiên bản | v1.0.0 (`models/v1.0.0/`) |
| Kiến trúc | ResNet-18 `resnet18.a1_in1k` từ `timm`, khởi tạo ImageNet, fine-tune toàn bộ (D4) |
| Đầu vào | Ảnh xám, resize về 224×224 bằng cách kéo giãn (`squash`), nhân 3 kênh, chuẩn hóa ImageNet |
| Huấn luyện | Cross-entropy có trọng số lớp, AdamW lr 1e-4, dừng sớm theo AUC val; epoch tốt nhất là 9 |
| Ngưỡng quyết định | 0,2313, chọn bằng chỉ số Youden trên val (D5). Xác suất ≥ ngưỡng thì là PNEUMONIA |
| Nguồn gốc | Lần chạy `20261026_190912_train_seed42`, commit `ad0ab9d`, seed 42, torch 2.14.1, timm 1.0.30 |
| Tính toán | Float32 đầy đủ, tắt TF32, để GPU và CPU cho cùng kết quả |

## Dữ liệu

- Bộ Kermany (Guangzhou Women and Children's Medical Center), bản Hugging Face `hf-vision/chest-xray-pneumonia`, revision `c1a67c1`.
- Trẻ em 1 đến 5 tuổi, một bệnh viện.
- Train 4.219 ảnh, val 704 ảnh, test 624 ảnh (234 NORMAL, 390 PNEUMONIA).
- Val tách từ train gốc, theo nhóm: ảnh cùng bệnh nhân hoặc trùng nhau (pHash) nằm cùng một tập.
- Test giữ nguyên như bản công bố.
- 283 ảnh RGB bị cắt sát phổi, toàn là PNEUMONIA, bị loại khỏi train/val (D3).

## Kết quả

Khoảng tin cậy 95% tính bằng bootstrap phân tầng, 1.000 vòng (D7). Ngưỡng chấp nhận đã chốt trước khi đo, ở D6.

| Chỉ số | Val | Test | Ngưỡng D6 (test) | Kết quả |
| :-- | :-- | :-- | :-- | :-- |
| AUC | 0,998 [0,997; 1,000] | 0,959 [0,941; 0,973] | ≥ 0,93 | Đạt |
| Độ nhạy | 0,980 [0,967; 0,992] | 0,995 [0,987; 1,000] | ≥ 0,90 | Đạt |
| Độ đặc hiệu | 0,995 [0,984; 1,000] | 0,735 [0,675; 0,791] | ≥ 0,75 | **Không đạt** |

- Ma trận nhầm lẫn trên test: TN 172, FP 62, FN 2, TP 388.
- Mô hình gần như không bỏ sót viêm phổi (2 trên 390 ca). Đổi lại, nó báo nhầm 62 trên 234 ảnh bình thường (26,5%).
- Test chỉ được đánh giá một lần. Ngưỡng không chọn lại trên test.
- Hình trong `reports/figures/`: `v1.0.0_roc.png`, `v1.0.0_confusion.png`, `v1.0.0_prob_normal.png` (thang log).

### Độ ổn định giữa các seed

Cùng cấu hình, train lại với 3 seed (42 là v1.0.0, thêm 0 và 1). Mỗi run đo trên test với ngưỡng chọn trên val của chính nó (D8). Chi tiết ở `reports/evaluation/multi_seed/`.

| Seed | AUC | Độ nhạy | Độ đặc hiệu |
| :-- | --: | --: | --: |
| 42 (v1.0.0) | 0,959 | 0,995 | 0,735 |
| 0 | 0,974 | 0,997 | 0,645 |
| 1 | 0,965 | 0,997 | 0,628 |
| Trung bình ± độ lệch chuẩn | 0,966 ± 0,007 | 0,997 ± 0,001 | 0,670 ± 0,057 |

- Độ lệch chuẩn AUC 0,007 ≤ 0,01: **đạt** tiêu chí ổn định của D6.
- AUC và độ nhạy ổn định. Độ đặc hiệu thì không: chênh tới 0,11 giữa các seed, và cả ba đều dưới 0,75.
- v1.0.0 có độ đặc hiệu cao nhất trong ba seed. Con số 0,735 vì vậy lạc quan hơn mức thường gặp; mức kỳ vọng khi train lại là khoảng 0,67.
- Các seed mới chỉ dùng để đo dao động, không dùng để chọn mô hình. v1.0.0 giữ nguyên.
- Chỉ có 3 run, nên độ lệch chuẩn ước lượng còn thô.

## Phân tích lỗi

Chi tiết ở `reports/evaluation/v1.0.0/error_analysis.md`. Phân tích này chỉ để mô tả, không dùng để chỉnh mô hình.

1. **Lỗi không nằm sát ngưỡng.** Trong 62 ảnh báo nhầm, 32 ảnh có xác suất ≥ 0,9. Chỉ 15 ảnh nằm giữa ngưỡng và 0,5. Vì vậy dời ngưỡng một chút không sửa được phần lớn lỗi.
2. **Ảnh NORMAL của test khác ảnh NORMAL của val.**
   - Trên val, 0,5% ảnh NORMAL vượt ngưỡng. Trên test, con số này là 26,5%.
   - Ảnh NORMAL của test rộng hơn: tỉ lệ rộng/cao trung vị là 1,35, so với 1,22 ở val.
3. **Kích thước ảnh gốc gắn với nhãn.**
   - Chỉ riêng chiều cao ảnh gốc đã phân biệt được hai nhãn trên train, với AUC 0,91 (ảnh PNEUMONIA thấp hơn). Trên test là 0,92.
   - Train và val đã được lọc theo D3, còn test thì không, nên so sánh giữa các tập chỉ mang tính tham khảo.
4. **Ảnh NORMAL nhỏ bị báo nhầm nhiều nhất.**
   - Chia ảnh NORMAL của test thành ba nhóm bằng nhau theo chiều cao ảnh gốc:

     | Nhóm | Chiều cao (px) | Tỉ lệ báo nhầm |
     | :-- | :-- | --: |
     | Nhỏ | 496 đến 1.113 | 57,7% |
     | Giữa | 1.122 đến 1.504 | 16,7% |
     | Lớn | 1.505 đến 2.713 | 5,1% |

   - Xác suất PNEUMONIA giảm khi ảnh cao hơn (tương quan hạng −0,51).
   - Tỉ lệ rộng/cao chỉ có tương quan với xác suất trong nhóm ảnh nhỏ (0,41). Ở hai nhóm còn lại, tương quan gần 0. Vậy yếu tố chính là ảnh nhỏ, không phải ảnh rộng.
   - Ảnh NORMAL nhỏ nhất của test cao 496 px, trong khi của val là 769 px. Test có những ảnh NORMAL nhỏ hơn mọi ảnh NORMAL mà mô hình đã gặp.
   - Xem trực tiếp: ảnh bị báo nhầm thường bị cắt sát lồng ngực, ít thấy cổ, vai, bụng.
   - 62 ảnh bị báo nhầm thuộc 62 bệnh nhân khác nhau, nên lỗi không dồn vào vài bệnh nhân.
5. **Đây là mối liên hệ, chưa phải nguyên nhân.**
   - Một khả năng: mô hình đã học một phần "ảnh nhỏ, cắt sát" như dấu hiệu của PNEUMONIA. Đây là rủi ro D3 đã nêu.
   - Một khả năng khác: kích thước ảnh đi kèm tuổi trẻ (trẻ nhỏ hơn thì phim nhỏ hơn), và trẻ nhỏ hơn có thể trông khác trên phim. Dữ liệu không có tuổi, nên không tách được hai khả năng.
   - Một số ảnh cũng có thể bị gán nhãn sai. Chưa có bác sĩ xem lại.
   - Muốn kiểm chứng cần thí nghiệm riêng. Nếu thí nghiệm đó dẫn tới train lại, test hiện tại không còn là đánh giá độc lập cho mô hình mới (D5, D6).

## Giới hạn

- Trẻ em 1 đến 5 tuổi, một bệnh viện, một nguồn dữ liệu. Không áp dụng được cho người lớn hay bệnh viện khác.
- Độ đặc hiệu giảm mạnh khi ảnh khác phân bố: 0,995 trên val, 0,735 trên test (trung bình 3 seed: 0,670 ± 0,057).
- Kích thước ảnh gốc gắn với nhãn trong dữ liệu. Mô hình báo nhầm nhiều hơn hẳn trên ảnh NORMAL nhỏ.
- Không có mã bệnh nhân thật. Mã được suy ra từ tên file (D1). pHash chỉ giảm rò rỉ, không loại bỏ hoàn toàn.
- Test có 6 cặp ảnh trùng, được tính hai lần (D2).
- Bootstrap coi mỗi ảnh là độc lập. Khoảng tin cậy không tính dao động giữa các seed train.
- Chỉ dùng cho nghiên cứu. Không thay thế bác sĩ.
