# Thẻ dữ liệu: X-quang ngực trẻ em (Kermany)

Bản đầu, 29/10/2026. Số liệu lấy từ `reports/data_quality/`. Mã quyết định (D1, D2...) trỏ tới `docs/decisions.md`.

## Nguồn

| Mục | Giá trị |
| :-- | :-- |
| Bộ dữ liệu gốc | Kermany và cộng sự (2018), Guangzhou Women and Children's Medical Center, Quảng Châu |
| Bản dùng trong đề tài | Hugging Face `hf-vision/chest-xray-pneumonia` |
| Revision ghim | `c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc` (`configs/data.yaml`) |
| Giấy phép | CC BY 4.0. Dùng lại phải ghi nguồn (xem `docs/references.md`) |
| Dung lượng tải | Khoảng 1,2 GB (file parquet) |
| Lưu trữ | Không commit vào git. Tải lại bằng `uv run python -m cxr.data.download` |

## Nội dung

- Ảnh X-quang ngực thẳng (trước sau) của trẻ em 1 đến 5 tuổi, chụp trong chăm sóc thường quy.
- Một bệnh viện, một nguồn. Không có tuổi, giới, thiết bị chụp hay mã bệnh nhân.
- Hai nhãn: `0` NORMAL (bình thường), `1` PNEUMONIA (viêm phổi do vi khuẩn hoặc virus).
- Toàn bộ 5.856 ảnh là JPEG, không ảnh nào hỏng, không nhãn nào sai định dạng.

Số ảnh theo split gốc của bản công bố:

| Split gốc | NORMAL | PNEUMONIA | Tổng | Tỉ lệ PNEUMONIA |
| :-- | --: | --: | --: | --: |
| train | 1.341 | 3.875 | 5.216 | 74,29% |
| validation | 8 | 8 | 16 | 50,00% |
| test | 234 | 390 | 624 | 62,50% |

Kích thước ảnh gốc rất khác nhau:

| | Nhỏ nhất | p05 | Trung vị | p95 | Lớn nhất |
| :-- | --: | --: | --: | --: | --: |
| Chiều rộng (px) | 384 | 848 | 1.281 | 1.966 | 2.916 |
| Chiều cao (px) | 127 | 512 | 888 | 1.697 | 2.713 |

## Xử lý trong pipeline

1. **Kiểm tra** (`validate.py`). Gắn cờ ảnh nhỏ (cạnh ngắn < 256 px), ảnh quá dẹt (tỉ lệ > 2,5), ảnh lưu dạng RGB. Kết quả: 58 ảnh nhỏ, 27 ảnh quá dẹt, 283 ảnh RGB.
2. **Mã bệnh nhân** (`dedup.py`, D1). Suy ra từ tên file gốc: `personN_bacteria`, `personN_virus`, `IM-XXXX`, `NORMAL2-IM-XXXX`. Được 4.097 bệnh nhân. Không bệnh nhân nào nằm ở cả train và test.
3. **Ảnh trùng** (`dedup.py`, D2). pHash 256 bit, ngưỡng Hamming 20. Có 34 cặp trùng, đều là bản sao giống từng byte: 28 cặp trong train, 6 cặp trong test. Không cặp nào giữa train và test.
4. **Loại ảnh khỏi train/val** (`split.py`, D2, D3).
   - 283 ảnh RGB. Đây cũng là toàn bộ ảnh nhỏ và ảnh quá dẹt của train. Tất cả là PNEUMONIA và bị cắt sát phổi, nên có nguy cơ thành đường tắt cho mô hình.
   - 26 bản sao thừa của các cặp trùng.
5. **Chia tập** (`split.py`). Gộp 16 ảnh validation gốc vào train. Tách val khoảng 15% (cấu hình 0,15, thực tế 14,3%) bằng `StratifiedGroupKFold`: phân tầng theo nhãn, ảnh cùng nhóm (cùng bệnh nhân hoặc trùng nhau) luôn nằm cùng một tập. Test giữ nguyên 624 ảnh, không lọc.

Kết quả chia tập (`data/processed/splits.csv`):

| Tập | NORMAL | PNEUMONIA | Tổng | Số nhóm | Tỉ lệ PNEUMONIA |
| :-- | --: | --: | --: | --: | --: |
| train | 1.155 | 3.064 | 4.219 | 2.997 | 72,62% |
| val | 193 | 511 | 704 | 501 | 72,59% |
| test | 234 | 390 | 624 | 427 | 62,50% |

- Không nhóm nào nằm ở hơn một tập. `split.py` kiểm tra và dừng nếu vi phạm.
- Thống kê tham chiếu cho giám sát (`data/reference/stats.json`) chỉ tính trên train.

## Rủi ro và giới hạn

- **Mã bệnh nhân là suy luận.** Dựa trên quy ước đặt tên file, không có dữ liệu bệnh nhân gốc để xác nhận (D1).
- **pHash chỉ giảm rò rỉ.** Hai ảnh khác nhau của cùng một người mà tên file không lộ ra thì vẫn có thể nằm ở hai tập.
- **Kích thước ảnh gắn với nhãn.** Ngay cả sau khi lọc, ảnh NORMAL có độ phân giải cao hơn hẳn: chiều cao trung vị khoảng 1.330 px, so với khoảng 800 px ở PNEUMONIA. Chỉ riêng chiều cao đã phân biệt hai nhãn với AUC khoảng 0,91. Mô hình có thể học một phần đặc điểm này (xem `docs/model_card.md`).
- **Test khác phân bố với train.** Tỉ lệ PNEUMONIA 62,5% so với 72,6%. Ảnh NORMAL của test rộng hơn ảnh NORMAL của val (tỉ lệ rộng/cao trung vị 1,35 so với 1,22). Test có ảnh NORMAL thấp tới 496 px, trong khi ảnh NORMAL thấp nhất của val là 769 px.
- **6 cặp trùng trong test** được giữ như bản công bố, nên 6 ảnh được tính hai lần khi đo (D2).
- **Phạm vi hẹp.** Trẻ 1 đến 5 tuổi, một bệnh viện. Không đại diện cho người lớn, bệnh viện hay thiết bị khác.
- **Chưa có bác sĩ xem lại nhãn.** Một số ảnh có thể bị gán nhãn sai.

## Tái tạo

```bash
uv run python scripts/run_pipeline.py --version v1.1.0 --to reference_stats
```

Lệnh trên chạy lại 5 bước dữ liệu và ghi đè đầu ra của chúng: `reports/data_quality/`, `data/reference/stats.json` (hai mục này có trong git), `data/interim/` và `data/processed/`. Nếu dữ liệu tái lập đúng, `git diff` không đổi số liệu và `splits.csv` có cùng mã hash ghi trong `models/v1.0.0/metadata.json`.
