# Báo Cáo Kết Quả Thực Nghiệm (ResNet Backbone)

Báo cáo dưới đây tổng hợp kết quả đánh giá (Evaluation) của các phương pháp khác nhau trên 3 bộ dữ liệu: **OwnOriginalDataset**, **IITD**, và **Tongji**. Các phương pháp đều sử dụng chung nền tảng trích xuất đặc trưng là ResNet.

## Bảng Tổng Hợp Kết Quả

| Bộ Dữ Liệu (Dataset) | Phương Pháp (Method) | Độ chính xác Rank-1 (Matching Accuracy) | Tỉ lệ lỗi EER |
| :--- | :--- | :--- | :--- |
| **OwnOriginalDataset** | Dùng trực tiếp Contrastive ArcFace | 99.43% | 0.08% |
| **OwnOriginalDataset** | Dùng thêm SVM | 98.70% | N/A |
| **IITD** | Dùng trực tiếp Contrastive ArcFace | 42.61% | 17.69% |
| **IITD** | Dùng thêm SVM | 22.17% | N/A |
| **IITD** | Optimize r (Input Space: Z + r) | 49.13% | 14.12% |
| **IITD** | Optimize r (Projected Space) | 46.09% | 13.48% |
| **Tongji** | Dùng trực tiếp Contrastive ArcFace | 33.50% | 17.42% |
| **Tongji** | Dùng thêm SVM | 34.83% | N/A |
| **Tongji** | Optimize r (Input Space: Z + r) | 84.00% | 3.54% |
| **Tongji** | Optimize r (Projected Space) | 84.00% | 3.00% |

## Nhận xét nhanh
1. **Trên dữ liệu tự thu thập (OwnOriginalDataset):** Mô hình ResNet+ArcFace trực tiếp đã cho kết quả gần như tuyệt đối (Rank-1 > 99%). Việc dùng thêm SVM không đem lại hiệu quả tốt hơn trong trường hợp này.
2. **Trên dữ liệu IITD:** Các phương pháp baseline thuần túy (ArcFace, SVM) bị tụt độ chính xác khá nhiều (khoảng 22% - 42%). Tuy nhiên, khi áp dụng cơ chế **Optimize r** (bù trừ nhiễu / đặc trưng), hiệu năng đã tăng lên đáng kể (Rank-1 đạt 49.13%, EER giảm còn ~13.48%).
3. **Trên dữ liệu Tongji:** Đây là nơi thấy rõ nhất sức mạnh của phương pháp Optimize r. Trong khi ArcFace thuần hay SVM chỉ đạt Rank-1 khoảng 33-34%, phương pháp Optimize r (ở cả Z+r và Projected Space) đã đẩy Rank-1 lên mức **84.00%** và giảm EER xuống chỉ còn **3.00%**. 

*(Kết quả được trích xuất từ các file log `eval_results.txt` trong các thư mục baseline tương ứng).*
