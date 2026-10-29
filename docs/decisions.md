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

## D5. Chọn ngưỡng quyết định bằng chỉ số Youden trên val (26/10/2026)

**Bối cảnh.** Mô hình trả xác suất PNEUMONIA. Cần một ngưỡng để ra nhãn, và ngưỡng quyết định độ nhạy, độ đặc hiệu báo cáo trên test.

**Quyết định.** Ngưỡng là giá trị làm J = độ nhạy + độ đặc hiệu − 1 lớn nhất trên **val**; nếu bằng nhau, lấy ngưỡng cao nhất. Ảnh có xác suất ≥ ngưỡng là PNEUMONIA. Ngưỡng này áp nguyên lên test, không chọn lại trên test. Cấu hình: `configs/eval.yaml`, `threshold.method: youden`.

**Lý do.** Đề tài chỉ dùng cho nghiên cứu, không có chi phí lâm sàng cụ thể để ưu tiên độ nhạy hay độ đặc hiệu. Youden coi hai loại lỗi ngang nhau, phổ biến và dễ giải thích. Phương án còn lại (độ nhạy tối thiểu, vd 0,95) cần một mục tiêu có căn cứ, hiện chưa có.

**Hệ quả.** `train.py` ghi ngưỡng và chỉ số trên val vào `val_metrics.json` của mỗi lần chạy. Có thể chọn lại ngưỡng từ `val_predictions.csv` mà không train lại, nhưng mọi thay đổi phải ghi vào đây trước khi đo trên test.

## D6. Ngưỡng chấp nhận cho từng mục tiêu (26/10/2026)

**Bối cảnh.** Ngưỡng phải chốt **trước khi** đo mô hình chính và trước khi xem bất kỳ kết quả nào trên test. Căn cứ duy nhất là baseline trên val (`runs/20261026_171211_train_baseline_seed42/`): backbone ResNet-18 đóng băng, 20 epoch, AUC val 0,9909; ở ngưỡng Youden 0,595: độ nhạy 0,945, độ đặc hiệu 0,979. Baseline chưa hội tụ hẳn (AUC còn tăng ~0,0002/epoch ở cuối). Chưa ai nhìn vào test.

**Quyết định.** Một tiêu chí "đạt" khi ước lượng điểm ≥ ngưỡng. Mọi chỉ số trên test báo cáo kèm khoảng tin cậy bootstrap 95%.

1. Đánh giá trung thực (mô hình chính, trên test, ngưỡng quyết định chọn trên val theo D5):

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| AUC test | ≥ 0,93 |
| Độ nhạy test | ≥ 0,90 |
| Độ đặc hiệu test | ≥ 0,75 |
| Chọn mô hình v1.0.0 | Mô hình chính nếu AUC val > 0,9909 (baseline); nếu không, dùng baseline. Chỉ dựa trên val |
| Ổn định giữa 3 seed | Độ lệch chuẩn AUC test ≤ 0,01 |

2. Tái lập được:

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| Dữ liệu chạy lại trên máy sạch | Train/val/test đúng 4.219 / 704 / 624 ảnh; `splits.csv` giống từng dòng |
| AUC test chạy lại | Lệch ≤ max(2 × độ lệch chuẩn giữa các seed; 0,005) |

3. Dữ liệu tin cậy:

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| `group_id` nằm ở hơn một tập | 0 |
| Ảnh trùng hoặc gần trùng giữa train/val và test | 0 |
| Báo cáo số ảnh và ảnh bị loại | Có, trong `reports/data_quality/` |

4. Triển khai được (API trên CPU):

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| Độ trễ p95 của `/predict` | ≤ 500 ms mỗi ảnh |
| Chênh xác suất giữa API và dự đoán offline cùng ảnh | < 1e-5 |
| Đầu vào hỏng hoặc sai định dạng | 100% trả lỗi 4xx, có test |
| Kích thước Docker image | ≤ 2 GB |

5. Chất lượng kỹ thuật:

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| pytest | 100% qua |
| ruff | 0 lỗi |
| Test unit | Mọi module trong `src/cxr/` có ít nhất một test |

6. Giám sát đầu vào:

| Tiêu chí | Ngưỡng |
| :-- | :-- |
| Cảnh báo nhầm trên ảnh test sạch | ≤ 5% |
| Phát hiện ở mức suy giảm mạnh nhất | ≥ 90% cho mỗi loại (mờ, nhiễu, tối, nén JPEG); các mức nhẹ hơn chỉ báo cáo |

**Lý do.**
- Val tách từ cùng nguồn với train. Test của bộ Kermany được công bố riêng và khác phân bố (62,5% so với 72,6% PNEUMONIA), nên chỉ số trên test dự kiến thấp hơn val. Các ngưỡng hiệu năng chừa khoảng hụt: AUC 6 điểm, độ nhạy 4,5 điểm, độ đặc hiệu nhiều hơn vì đây là điểm yếu thường gặp trên test này.
- Cỡ mẫu test giới hạn độ chính xác: khoảng tin cậy 95% rộng khoảng ±0,02 cho độ nhạy (390 ca) và ±0,05 cho độ đặc hiệu (234 ca). Vì vậy dùng ước lượng điểm, không đòi cận dưới khoảng tin cậy vượt ngưỡng.
- Độ trễ 500 ms là ước lượng an toàn cho ResNet-18 trên CPU cộng thời gian giải mã ảnh lớn, chưa đo.
- Ngưỡng cảnh báo nhầm 5% buộc `drift.py` giữ ít chỉ số hoặc khoảng tham chiếu đủ rộng, vì báo nhầm cộng dồn qua các chỉ số.

**Hệ quả.** Từ đây không đổi các ngưỡng này sau khi đã thấy kết quả test. Nếu buộc phải đổi, ghi một mục mới kèm lý do và nêu rõ trong báo cáo. Bảng này cần chép vào đề cương (mục 2.2).

**Giới hạn.** Ngưỡng AUC 0,93 và độ đặc hiệu 0,75 dựa trên nhận định chung rằng test của bộ dữ liệu này khó hơn val, không dựa trên số đo test.

## D7. Cách tính khoảng tin cậy bootstrap (26/10/2026)

**Bối cảnh.** D6 yêu cầu mọi chỉ số trên test kèm khoảng tin cậy 95%. Test có 624 ảnh: 234 NORMAL, 390 PNEUMONIA. Cần chốt cách tính trước khi đo trên test.

**Quyết định.** `src/cxr/evaluation/bootstrap.py`, cấu hình ở `configs/eval.yaml`, mục `bootstrap`:
- 1.000 vòng, độ tin cậy 0,95, seed 0.
- Mỗi vòng lấy mẫu có hoàn lại **riêng từng lớp**, giữ nguyên số ảnh mỗi lớp (bootstrap phân tầng).
- Ngưỡng quyết định giữ cố định (ngưỡng chọn trên val theo D5), không chọn lại trong từng vòng.
- Khoảng tin cậy là khoảng phân vị: phân vị 2,5% và 97,5% của 1.000 giá trị.
- Giá trị báo cáo là chỉ số trên toàn bộ test, không phải trung bình các vòng.

**Lý do.**
- Tỉ lệ lớp của test cố định theo cách bộ dữ liệu được công bố, không do ngẫu nhiên. Phân tầng giữ đúng điều này, và mỗi vòng luôn có đủ hai lớp nên AUC luôn tính được.
- Độ nhạy chỉ phụ thuộc ảnh PNEUMONIA, độ đặc hiệu chỉ phụ thuộc ảnh NORMAL. Khi phân tầng, khoảng tin cậy của hai chỉ số này gần với khoảng nhị thức thông thường, dễ kiểm tra.
- Giữ ngưỡng cố định vì khi triển khai, ngưỡng cũng cố định. Khoảng tin cậy chỉ phản ánh dao động do mẫu test, không phản ánh dao động do chọn ngưỡng.
- Khoảng phân vị đơn giản, phổ biến. Với 1.000 vòng, sai số của cận khoảng do số vòng gây ra nhỏ hơn nhiều so với độ rộng khoảng.

**Giới hạn.** Bootstrap coi mỗi ảnh độc lập. Bộ dữ liệu không có mã bệnh nhân, nên ảnh cùng bệnh nhân có thể làm khoảng tin cậy hẹp hơn thực tế. Khoảng này không phản ánh dao động giữa các seed train; dao động đó đo riêng bằng nhiều seed (D6).

Khi một chỉ số đúng bằng 0 hoặc 1 trên toàn bộ test (vd không ảnh NORMAL nào bị báo nhầm), mọi vòng cho cùng giá trị và khoảng tin cậy rộng 0. Trường hợp này báo cáo là "bootstrap không ước lượng được", không phải "không có sai số".

## D8. Cách đo độ ổn định giữa các seed (27/10/2026)

**Bối cảnh.** D6 yêu cầu độ lệch chuẩn AUC test giữa 3 seed ≤ 0,01. Tiêu chí tái lập trên máy sạch cũng dùng con số này. Cần chốt cách đo trước khi xem kết quả test của các seed mới.

**Quyết định.** `src/cxr/evaluation/multi_seed.py`, ngưỡng ở `configs/eval.yaml`, mục `multi_seed`:
- 3 seed: 42 (run `20261026_190912_train_seed42`, nguồn của v1.0.0, dùng lại, không train lại), 0 và 1 (train mới với `configs/train.yaml`, chỉ đổi `--seed`).
- Mỗi run được đo trên test với ngưỡng quyết định của chính nó, chọn trên val theo D5.
- Báo cáo trung bình và độ lệch chuẩn mẫu (ddof=1) của AUC, độ nhạy, độ đặc hiệu. Chỉ AUC có ngưỡng chấp nhận.
- Trước khi đo test, module kiểm tra: seed khác nhau, cùng hash `splits.csv`, `config.json` giống nhau trừ `train.seed`. Phần `eval` chỉ so `threshold`, vì khi train chỉ dùng mục này; các mục `bootstrap`, `acceptance` được thêm sau run seed 42.
- Kết quả ghi một lần vào `reports/evaluation/multi_seed/`, không ghi đè.

**Lý do.** Mỗi seed đi đúng quy trình như v1.0.0, nên độ lệch chỉ phản ánh dao động do seed. Code huấn luyện, cấu hình và `splits.csv` giống nhau giữa ba run (đã so `git diff ad0ab9d eb7c355` và hash `splits.csv`). Độ lệch chuẩn mẫu là cách tính thông thường khi chỉ có vài lần chạy.

**Hệ quả.** Các seed mới không dùng để chọn mô hình. v1.0.0 giữ nguyên, kể cả khi một seed khác có AUC test cao hơn. Với 3 run, độ lệch chuẩn ước lượng còn rất thô; cần nêu điều này trong báo cáo.

## D9. Chỉ số và ngưỡng giám sát chất lượng ảnh đầu vào (27/10/2026)

**Bối cảnh.** API phải trả cờ chất lượng ảnh. D6 yêu cầu báo nhầm ≤ 5% trên ảnh test sạch và phát hiện ≥ 90% ở mức suy giảm mạnh nhất của mỗi loại (mờ, nhiễu, tối, nén JPEG). Ngưỡng chọn trên val, chưa nhìn test.

**Quan sát (val, 704 ảnh).** Ba chỉ số cũ (độ sáng, tương phản, độ mờ) đo trên ảnh 224×224 phát hiện mờ và tối 100%, nhưng nhiễu (độ lệch chuẩn 40) chỉ 42% và JPEG chất lượng 5 chỉ 2%. Thu nhỏ khoảng 5 lần làm trung bình hóa nhiễu và xóa khối 8×8 của JPEG. Kích thước ảnh gốc báo nhầm nhiều (chiều cao một mình 2,1%) và gắn với nhãn.

**Quyết định.**
- Thêm 2 chỉ số vào `src/cxr/monitoring/image_stats.py`: `noise` (ước lượng nhiễu Immerkær, trên ảnh 224×224) và `blockiness` (tỉ lệ chênh cường độ tại biên khối 8 px so với chỗ khác, trên ảnh gốc). Tính lại `data/reference/stats.json`; các chỉ số cũ không đổi.
- `src/cxr/monitoring/drift.py` kiểm 5 chỉ số, mỗi chỉ số theo hướng suy giảm đẩy nó đi. Khoảng là phân vị của train, cấu hình ở `configs/monitor.yaml`:

| Chỉ số | Thấp | Cao |
| :-- | :-- | :-- |
| Độ sáng | p0,5 | p99,5 |
| Tương phản | p0,5 | p99,5 |
| Độ mờ | p0,5 | không kiểm |
| `noise` | không kiểm | p99,5 |
| `blockiness` | không kiểm | p99,5 |

- Không kiểm kích thước ảnh. Một chỉ số ra ngoài khoảng thì ảnh bị "cảnh báo". Cảnh báo không chặn dự đoán.

**Kết quả hiệu chỉnh trên val.** Báo nhầm 3,55% (từng chỉ số 0,4% đến 1,0%). Trên 60 ảnh val suy giảm mạnh (mờ bán kính 8, nhiễu 40, tối ×0,3, JPEG chất lượng 10): phát hiện 100% mỗi loại.

**Lý do.** Kiểm một phía giảm báo nhầm cộng dồn mà không mất khả năng phát hiện. `blockiness` chỉ kiểm phía cao vì ảnh train vốn là JPEG (trung vị 1,61); ảnh PNG cho khoảng 1,0, không phải lỗi.

**Giới hạn.** Test khác phân bố với train, nên báo nhầm trên test có thể cao hơn 3,55%; lề còn 1,45 điểm. Kết quả chính thức đo trên test ở giai đoạn 6, với các mức suy giảm chốt trước trong `degrade.py`. Hiệu chỉnh trên val dùng một script thăm dò, chưa nằm trong repo.

## D10. Các mức suy giảm nhân tạo để đánh giá giám sát (29/10/2026)

**Bối cảnh.** D6 yêu cầu báo nhầm ≤ 5% trên ảnh test sạch và phát hiện ≥ 90% ở mức suy giảm mạnh nhất của mỗi loại. D9 đã chốt chỉ số và ngưỡng giám sát trên val. Các mức suy giảm phải chốt trước khi đo trên test.

**Quyết định.** `src/cxr/monitoring/degrade.py`, cấu hình ở `configs/monitor.yaml`, mục `degradations`. Mỗi loại 4 mức, từ nhẹ đến mạnh. Mức cuối là mức mạnh nhất, dùng để xét đạt hoặc không đạt:

| Loại | Tham số | Các mức |
| :-- | :-- | :-- |
| Mờ | bán kính Gaussian (px, ảnh gốc) | 1, 2, 4, 8 |
| Nhiễu | độ lệch chuẩn nhiễu Gaussian (mức xám 0 đến 255, ảnh gốc) | 5, 10, 20, 40 |
| Tối | hệ số nhân giá trị điểm ảnh | 0,9; 0,7; 0,5; 0,3 |
| Nén JPEG | chất lượng JPEG | 50, 30, 20, 10 |

- Suy giảm áp lên ảnh gốc ở dạng xám 8 bit, trước mọi bước thu nhỏ, giống ảnh thật gửi tới API.
- Nhiễu có seed theo (`noise_seed`, thứ tự ảnh), nên chạy lại cho cùng kết quả.
- `src/cxr/monitoring/evaluate_monitoring.py` đưa mỗi ảnh qua `Predictor` của API, nên kiểm tra chất lượng và xác suất giống hệt `/predict`. Kết quả ghi vào `reports/monitoring/v1.0.0/`. Test chỉ đo một lần, không ghi đè.
- Kèm theo, đo AUC, độ nhạy, độ đặc hiệu của mô hình trên từng mức. Các số này chỉ để báo cáo, không có ngưỡng.

**Lý do.** Mức mạnh nhất trùng với mức đã dùng khi hiệu chỉnh trên val ở D9 (mờ 8, nhiễu 40, tối 0,3, JPEG chất lượng 10). Con số "JPEG chất lượng 5" trong phần quan sát của D9 là phép thử với ba chỉ số cũ, không phải mức hiệu chỉnh. Các mức nhẹ hơn cho thấy giám sát bắt đầu phát hiện từ đâu.

**Giới hạn.** Ảnh mờ, nhiễu, tối được đưa thẳng vào `Predictor` dưới dạng ảnh trong bộ nhớ, không mã hóa lại thành file. Một ảnh thật gửi lên dạng JPEG sẽ có thêm một lần nén.

## D11. Cách đo kích thước Docker image (29/10/2026)

**Bối cảnh.** D6 đặt ngưỡng "kích thước Docker image ≤ 2 GB" nhưng không nói cách đo. Image `cxr-api:v1.0.0` có ba con số: 1,5 GB hệ thống file giải nén (`du -sx /` trong container), 486 MB nội dung nén, 2,1 GB theo cột DISK USAGE của Docker Desktop (`docker image ls`, gồm cả bản nén và bản giải nén).

**Quyết định.** Đo theo con số Docker Desktop hiển thị: cột DISK USAGE của `docker image ls`, tương đương `docker image inspect --format '{{.Size}}'`. Đơn vị theo cách Docker hiển thị: 1 GB = 10⁹ byte (không phải 1 GiB = 2³⁰ byte).

**Lý do.** Đây là con số người dùng thấy trực tiếp khi mở Docker Desktop, không cần giải thích thêm.

**Hệ quả.** Cách đo được chốt **sau khi** đã thấy cả ba con số. Với cách đo này, image hiện tại 2,10 GB (2.104.404.707 byte) **không đạt** ngưỡng 2 GB. Ngưỡng 2 GB giữ nguyên. Muốn đạt phải làm image nhỏ lại, ví dụ bỏ các thư viện API không dùng.
