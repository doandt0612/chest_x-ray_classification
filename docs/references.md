# Tài liệu tham khảo

Trích dẫn theo APA 7th. Bản nháp ngày 29/10/2026.

> **Cần kiểm tra trước khi nộp.** Các mục dưới đây được viết từ trí nhớ, chưa đối chiếu với bản gốc. Hãy kiểm tra lại tên tác giả, năm, số trang và DOI của từng mục.

## Dữ liệu

Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled optical coherence tomography (OCT) and chest X-ray images for classification* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.2

hf-vision. (n.d.). *chest-xray-pneumonia* [Data set]. Hugging Face. Retrieved October 24, 2026, from https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia

- Revision dùng trong đề tài: `c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc`.
- Giấy phép: CC BY 4.0, theo thẻ dữ liệu trên Hugging Face.

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., ... Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell, 172*(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

## Mô hình

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Wightman, R. (2019). *PyTorch image models* [Computer software]. GitHub. https://doi.org/10.5281/zenodo.4414861

Wightman, R., Touvron, H., & Jégou, H. (2021). *ResNet strikes back: An improved training procedure in timm* (arXiv:2110.00476). arXiv. https://doi.org/10.48550/arXiv.2110.00476

- Trọng số `resnet18.a1_in1k` dùng trong đề tài theo công thức huấn luyện A1 của bài này.

## Đánh giá

Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

## Giám sát chất lượng ảnh

Immerkær, J. (1996). Fast noise variance estimation. *Computer Vision and Image Understanding, 64*(2), 300–302. https://doi.org/10.1006/cviu.1996.0060

Pech-Pacheco, J. L., Cristóbal, G., Chamorro-Martínez, J., & Fernández-Valdivia, J. (2000). Diatom autofocusing in brightfield microscopy: A comparative study. In *Proceedings of the 15th International Conference on Pattern Recognition* (Vol. 3, pp. 314–317). IEEE. https://doi.org/10.1109/ICPR.2000.903548

- Bài này dùng phương sai Laplacian để đo độ nét, cách `image_stats.py` đo độ mờ.

## Tài liệu hóa mô hình và dữ liệu

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM, 64*(12), 86–92. https://doi.org/10.1145/3458723

## Thư viện chính

| Thư viện | Dùng cho |
| :-- | :-- |
| PyTorch, torchvision | Huấn luyện và suy luận |
| timm | Backbone ResNet-18 tiền huấn luyện |
| scikit-learn | AUC, đường ROC, `StratifiedGroupKFold` |
| imagehash | pHash tìm ảnh trùng |
| Pillow, NumPy, pandas | Xử lý ảnh và bảng |
| huggingface_hub, pyarrow | Tải và đọc dữ liệu |
| FastAPI, Uvicorn | API |
| pytest, httpx, ruff | Kiểm thử và lint |

Phiên bản chính xác ghim trong `uv.lock`.

## Mã nguồn tham khảo

> **Cần bạn điền.** PLAN yêu cầu ghi rõ repo tham khảo và phần nào tham khảo từ đâu. Tôi không biết bạn đã tham khảo repo nào. Ghi theo mẫu:
>
> | Repo hoặc tài liệu | Phần tham khảo | File trong đề tài |
> | :-- | :-- | :-- |
> | | | |
