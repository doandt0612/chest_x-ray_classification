# Hướng dẫn demo

Demo gồm 5 bước: chạy API trong Docker, gửi ảnh sạch, gửi ảnh suy giảm, gửi file hỏng, đo độ trễ. Mất khoảng 10 phút. Đã chạy thử toàn bộ ngày 29/10/2026.

## Chuẩn bị

- Docker Desktop đang chạy. Trên Windows, bật tích hợp WSL cho distro đang dùng.
- Đã có model `models/v1.0.0/` và ảnh test `data/raw/images/test/`. Hai thư mục này không có trong git.
  - Ảnh test: tải lại bằng `uv run python -m cxr.data.download`.
  - Model: chép `models/v1.0.0/` từ máy đã train, hoặc từ bản sao lưu nếu có. Chạy lại pipeline sẽ train một model mới, không khôi phục đúng v1.0.0.
- Mọi lệnh chạy từ thư mục gốc của repo.

## 1. Build và chạy API

```bash
docker build -t cxr-api:v1.0.0 .
docker run --rm -d -p 8000:8000 --name cxr-api cxr-api:v1.0.0
curl http://localhost:8000/health
```

Kết quả mong đợi:

```json
{"status":"ok","model_version":"v1.0.0"}
```

## 2. Gửi một ảnh sạch

```bash
curl -F "file=@data/raw/images/test/test_00000.jpeg" http://localhost:8000/predict
```

Kết quả: nhãn `NORMAL`, xác suất PNEUMONIA khoảng 0,0016, `quality.status` là `ok`, không có cờ. Phản hồi luôn có dòng `"disclaimer": "Research use only. Not a medical device; not for clinical diagnosis."`.

## 3. Gửi ảnh suy giảm

Tạo 3 bản suy giảm của cùng ảnh trên bằng `src/cxr/monitoring/degrade.py`. Lưu dạng PNG để không thêm một lần nén JPEG.

```bash
uv run python -c "
import numpy as np
from PIL import Image
from cxr.monitoring.degrade import degrade
img = Image.open('data/raw/images/test/test_00000.jpeg')
for kind, level in [('dark', 0.3), ('blur', 8), ('noise', 40)]:
    degrade(img, kind, level, np.random.default_rng(0)).save(f'demo_{kind}.png')
"
curl -F "file=@demo_dark.png" http://localhost:8000/predict
curl -F "file=@demo_blur.png" http://localhost:8000/predict
curl -F "file=@demo_noise.png" http://localhost:8000/predict
```

Kết quả đã chạy thử:

| Ảnh | Nhãn | Xác suất PNEUMONIA | Chất lượng | Cờ |
| :-- | :-- | --: | :-- | :-- |
| Sạch | NORMAL | 0,0016 | ok | không có |
| Tối ×0,3 | **PNEUMONIA** | 0,9947 | warning | `brightness:low`, `contrast:low`, `blur:low` |
| Mờ bán kính 8 | NORMAL | 0,0021 | warning | `blur:low` |
| Nhiễu 40 | **PNEUMONIA** | 0,5514 | warning | `noise:high` |

Điểm cần nói khi demo:

- Cùng một ảnh bình thường, chỉ làm tối hoặc thêm nhiễu là mô hình đổi sang PNEUMONIA.
- API vẫn trả dự đoán, nhưng gắn cảnh báo chất lượng. Người dùng biết không nên tin kết quả này.
- Trên toàn bộ test, cảnh báo nhầm trên ảnh sạch là 2,9%. Ở mức suy giảm mạnh nhất, 100% ảnh bị cảnh báo (`reports/monitoring/v1.0.0/test_table.csv`).
- Giới hạn: ở các mức nhẹ, mô hình đã sai nhiều hơn trong khi phần lớn ảnh chưa bị cảnh báo. Ví dụ, tối ×0,7 làm độ đặc hiệu giảm từ 0,735 xuống 0,440, nhưng chỉ 32% ảnh bị cảnh báo.

## 4. Gửi file hỏng

```bash
echo "not an image" > bad.txt
curl -s -o /dev/null -w "%{http_code}\n" -F "file=@bad.txt" http://localhost:8000/predict
```

Kết quả: `400`. Các lỗi khác: file rỗng trả 400, file trên 10 MB trả 413, định dạng khác JPEG hoặc PNG trả 415, thiếu trường `file` trả 422.

## 5. Đo độ trễ

```bash
uv run python -m cxr.api.benchmark_latency --url http://localhost:8000 --out-dir demo_latency
```

`--out-dir demo_latency` để không ghi đè báo cáo chính thức `reports/latency/v1.0.0.json` đã commit.

Lệnh này gửi tuần tự 624 ảnh test, sau 20 request khởi động (D13). Kết quả trên máy Ryzen 7 7840H, WSL2: p50 41,9 ms, p95 84,7 ms. Ngưỡng D6 là p95 ≤ 500 ms.

## Dọn dẹp

```bash
docker stop cxr-api
rm -r demo_dark.png demo_blur.png demo_noise.png bad.txt demo_latency
```
