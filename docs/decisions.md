# Nhật ký quyết định

Mỗi mục ghi: bối cảnh, quyết định, lý do, hệ quả. Ngưỡng chấp nhận của mô hình phải được chốt ở đây **trước khi** đo mô hình chính.

---

## D1. Mã bệnh nhân lấy từ tên file, `personN` tách theo loại bệnh (25/10/2026)

**Bối cảnh.** Dataset không có cột mã bệnh nhân. Tên file gốc chứa mã: `personN_bacteria_*`, `personN_virus_*` (PNEUMONIA), `IM-XXXX-*`, `NORMAL2-IM-XXXX-*` (NORMAL).

**Quan sát.**
- Số NORMAL được đánh liên tục, không chồng nhau giữa các split (test `IM-1..111`, train `IM-115..766`).
- 170 số `personN` xuất hiện ở cả train và test, nhưng **không số nào cùng loại bệnh** ở hai bên; trong train, 808/1635 số `personN` lẫn cả bacteria và virus.

**Quyết định.** Mã bệnh nhân PNEUMONIA là `person{N}_{bacteria|virus}`; `IM-` và `NORMAL2-IM-` là hai dãy số riêng.

**Lý do.** Nếu `personN` là một người, gần như không thể có 0/170 trùng loại bệnh. Kết luận này là suy luận từ quy ước đặt tên, không có dữ liệu bệnh nhân gốc để xác nhận.

**Hệ quả.** 4.097 bệnh nhân; không bệnh nhân nào nằm ở cả train và test. `group_id` (dedup.py) gộp ảnh cùng bệnh nhân và ảnh trùng; split.py giữ mỗi nhóm trong một tập.

## D2. Ảnh trùng hoàn toàn (25/10/2026)

**Quan sát.** pHash 256 bit (`hash_size=16`) cho 34 cặp ở khoảng cách 0 (đều là bản sao giống từng byte), cặp gần nhất tiếp theo ở khoảng cách 36 và là hai bệnh nhân khác nhau. pHash 64 bit mặc định quá thô: ở khoảng cách 4 đã có 215 cặp, 40 cặp khác nhãn. Không có cặp trùng nào giữa train và test.

**Quyết định.** Ngưỡng `max_hamming=20`. Trong phần train/val: giữ một bản mỗi nhóm trùng (bỏ 26 ảnh). Test giữ nguyên 624 ảnh như bản công bố, dù có 6 cặp trùng.

**Hệ quả.** 6 ảnh test được tính hai lần khi đo metrics. Cần nêu điều này trong báo cáo.

## D3. Loại nhóm ảnh RGB bị cắt quanh phổi khỏi train/val (25/10/2026)

**Bối cảnh.** validate.py phát hiện 283 ảnh lưu dạng RGB (nội dung vẫn xám), **tất cả là PNEUMONIA** (143 bacteria, 140 virus), đều ở split train gốc. So với phần còn lại:

| | RGB | PNEUMONIA (xám) | NORMAL |
| :-- | --: | --: | --: |
| Số ảnh | 283 | 3.990 | 1.583 |
| Chiều cao trung vị (px) | 467 | 792 | 1.323 |
| Tỉ lệ rộng/cao trung vị | 1,70 | 1,48 | 1,23 |

Ảnh RGB bị cắt sát vùng phổi (không có vai, xương đòn, ký hiệu "R"); 59 ảnh nặng nhất chỉ là dải ngang (thấp nhất 384×127 px). Đây là toàn bộ ảnh bị gắn cờ `too_small`/`extreme_aspect` trong phần train. Không ảnh NORMAL nào có đặc điểm này.

**Quyết định.** Loại cả 283 ảnh khỏi train/val: `exclude_flags: [too_small, extreme_aspect, rgb_mode]` trong `configs/data.yaml`. Test không lọc.

**Lý do.** Đặc điểm "ảnh bị cắt, độ phân giải thấp" chỉ xuất hiện ở PNEUMONIA → mô hình có thể học đường tắt thay vì học bệnh, và resize 224×224 không xóa được khác biệt này. Test gần như không chứa loại ảnh này (0 RGB, 1 ảnh xám bị gắn cờ) nên AUC test không phát hiện được đường tắt. Mất ~7% ảnh PNEUMONIA của train; lớp này vốn chiếm đa số nên vẫn đủ dữ liệu.

**Hệ quả.** Train 4.219 ảnh (72,62% PNEUMONIA), val 704 (72,59%), test 624 (62,50%).

**Giới hạn còn lại.** Ngay cả ảnh xám, NORMAL vẫn có độ phân giải cao hơn PNEUMONIA rõ rệt (chiều cao trung vị ~1.330 so với ~800 px ở train). Đây là đặc điểm đã biết của dữ liệu Kermany, không loại bỏ được bằng cách lọc. Cần nêu trong `data_card.md` và `model_card.md`. Nhận định "cả 283 ảnh RGB đều bị cắt" dựa trên số liệu kích thước và một số ảnh đã xem trực tiếp, chưa xem hết từng ảnh.

## D4. Backbone ResNet-18 từ timm (26/10/2026)

**Bối cảnh.** PLAN.md đề xuất một backbone nhỏ, tiền huấn luyện ImageNet: ResNet-18 hoặc EfficientNet-B0. Không so sánh kiến trúc.

**Quyết định.** Dùng `resnet18` của thư viện `timm` (trọng số `resnet18.a1_in1k`), head mới `Linear(512, 2)`. Cấu hình ở `configs/train.yaml`, mục `model`.

**Lý do.** Khoảng 11,2 triệu tham số, chạy CPU nhanh, phù hợp API suy luận trên CPU. Mean/std của trọng số này trùng với chuẩn hóa ImageNet trong `configs/data.yaml`; `build.py` kiểm tra điều này khi tạo mô hình.

**Hệ quả.** Baseline: `freeze_backbone: true`, chỉ train head (1.026 tham số). Mô hình chính: fine-tune toàn bộ.
